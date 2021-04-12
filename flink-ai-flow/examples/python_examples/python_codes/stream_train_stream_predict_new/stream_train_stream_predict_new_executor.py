import os
import threading
import time
from typing import List
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from streamz import Stream
import ai_flow as af
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage
from python_ai_flow import FunctionContext, Executor, ExampleExecutor
from ai_flow.common.path_util import get_file_dir


def preprocess_data(x_data, y_data):
    random_state = check_random_state(0)
    permutation = random_state.permutation(x_data.shape[0])
    if y_data is None:
        return x_data[permutation]
    else:
        return x_data[permutation], y_data[permutation]


class ExampleTrainThread(threading.Thread):

    def __init__(self, stream_uri):
        super().__init__()
        self.stream_uri = stream_uri
        self.stream = Stream()

    def run(self) -> None:
        for i in range(0, 5):
            f = np.load(self.stream_uri)
            x_train, y_train = f['x_train'], f['y_train']
            f.close()
            self.stream.emit((x_train, y_train))
            time.sleep(30)


class TrainExampleReader(ExampleExecutor):

    def __init__(self):
        super().__init__()
        self.thread = None

    def setup(self, function_context: FunctionContext):
        stream_uri = function_context.node_spec.example_meta.stream_uri
        self.thread = ExampleTrainThread(stream_uri)
        self.thread.start()

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        return [self.thread.stream]


class TrainExampleTransformer(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        def transform(df):
            x_train, y_train = preprocess_data(df[0], df[1])
            x_train = x_train.reshape((x_train.shape[0], -1))
            return StandardScaler().fit_transform(x_train), y_train

        return [input_list[0].map(transform)]


class ModelTrainer(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))

        def train(df):
            # https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html
            clf = LogisticRegression(C=50. / 5000, penalty='l1', solver='saga', tol=0.1)
            x_train, y_train = df[0], df[1]
            clf.fit(x_train, y_train)
            model_path = get_file_dir(__file__) + '/saved_model'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_version = time.strftime("%Y%m%d%H%M%S", time.localtime())
            model_path = model_path + '/' + model_version
            dump(clf, model_path)
            model = function_context.node_spec.output_model
            print(model.name)
            print(model_version)

            af.register_model_version(model=model, model_path=model_path, current_stage=ModelVersionStage.GENERATED)
            return df

        def sink(df):
            pass

        input_list[0].map(train).sink(sink)
        return []


class EvaluateExampleReader(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        f = np.load(function_context.node_spec.example_meta.stream_uri)
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        return [[x_test, y_test]]


class EvaluateTransformer(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        x_test, y_test = preprocess_data(input_list[0][0], input_list[0][1])
        x_test = x_test.reshape((x_test.shape[0], -1))
        return [[StandardScaler().fit_transform(x_test), y_test]]


class ModelEvaluator(Executor):

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_path = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        self.model = af.get_latest_generated_model_version(function_context.node_spec.model.name)
        self.model_path = self.model.model_path
        self.model_version = self.model.version

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        x_evaluate, y_evaluate = input_list[0][0], input_list[0][1]
        clf = load(self.model_path)
        scores = cross_val_score(clf, x_evaluate, y_evaluate, cv=5)
        evaluate_artifact = af.get_artifact_by_name('evaluate_artifact2').stream_uri
        with open(evaluate_artifact, 'a') as f:
            f.write('model version[{}] scores: {}\n'.format(self.model_version, scores))
        return []


class ValidateExampleReader(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        f = np.load(function_context.node_spec.example_meta.stream_uri)
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        return [[x_test, y_test]]


class ValidateTransformer(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        x_test, y_test = preprocess_data(input_list[0][0], input_list[0][1])
        x_test = x_test.reshape((x_test.shape[0], -1))
        return [[StandardScaler().fit_transform(x_test), y_test]]


class ModelValidator(Executor):

    def __init__(self):
        super().__init__()
        self.model_name = None
        self.model_meta = None
        self.model_path = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        self.model_name = function_context.node_spec.model.name
        self.model_meta = af.get_latest_generated_model_version(self.model_name)
        self.model_path = self.model_meta.model_path
        self.model_version = self.model_meta.version

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        deployed_model_version = af.get_deployed_model_version(model_name=self.model_name)
        stream_uri = af.get_artifact_by_name('validate_artifact').stream_uri
        if deployed_model_version is None:
            with open(stream_uri, 'a') as f:
                f.write('generated model version[{}] scores: {}\n'.format(self.model_version, "Init"))
            af.update_model_version(model_name=self.model_name,
                                    model_version=self.model_version,
                                    current_stage=ModelVersionStage.DEPLOYED)
        else:
            x_validate, y_validate = input_list[0][0], input_list[0][1]
            clf = load(self.model_path)
            scores = cross_val_score(clf, x_validate, y_validate, scoring='precision_macro', cv=5)
            deployed_clf = load(deployed_model_version.model_path)
            deployed_scores = cross_val_score(deployed_clf, x_validate, y_validate, scoring='precision_macro')

            if np.mean(scores) > np.mean(deployed_scores):
                af.update_model_version(model_name=self.model_name,
                                        model_version=deployed_model_version.version,
                                        current_stage=ModelVersionStage.VALIDATED)
                af.update_model_version(model_name=self.model_name,
                                        model_version=self.model_version,
                                        current_stage=ModelVersionStage.DEPLOYED)
                with open(stream_uri, 'a') as f:
                    f.write('deployed model version[{}] scores: {}\n'.format(deployed_model_version.version,
                                                                             deployed_scores))
                    f.write('generated model version[{}] scores: {}\n'.format(self.model_version, scores))
        return []


class ExamplePredictThread(threading.Thread):

    def __init__(self, stream_uri):
        super().__init__()
        self.stream_uri = stream_uri
        self.stream = Stream()

    def run(self) -> None:
        for i in range(0, 5):
            f = np.load(self.stream_uri)
            x_test = f['x_test']
            f.close()
            self.stream.emit(x_test)
            print("### {} {}".format(self.__class__.__name__, "generate data flow"))
            time.sleep(30)


class PredictExampleReader(ExampleExecutor):

    def __init__(self):
        super().__init__()
        self.thread = None

    def setup(self, function_context: FunctionContext):
        stream_uri = function_context.node_spec.example_meta.stream_uri
        self.thread = ExamplePredictThread(stream_uri)
        self.thread.start()

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        return [self.thread.stream]


class PredictTransformer(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        def transform(df):
            x_test = preprocess_data(df, None)
            x_test = x_test.reshape((x_test.shape[0], -1))
            return StandardScaler().fit_transform(x_test)

        return [input_list[0].map(transform)]


class ModelPredictor(Executor):

    def __init__(self):
        super().__init__()
        self.model_name = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        self.model_name = function_context.node_spec.model.name
        while af.get_deployed_model_version(self.model_name) is None:
            time.sleep(5)
        print("### {} setup done for {}".format(self.__class__.__name__, function_context.node_spec.model.name))

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        def predict(df):
            x_test = df
            model_meta = af.get_deployed_model_version(self.model_name)
            model_path = model_meta.model_path
            clf = load(model_path)
            return model_meta.version, clf.predict(x_test)

        return [input_list[0].map(predict)]


class ExampleWriter(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        def write_example(df):
            with open(function_context.node_spec.example_meta.stream_uri, 'a') as f:
                f.write('model version[{}] predict: {}\n'.format(df[0], df[1]))
            return df

        def sink(df):
            pass

        input_list[0].map(write_example).sink(sink)
        return []
