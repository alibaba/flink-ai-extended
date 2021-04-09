import json
import os
import queue
import threading
import time

from ai_flow.common.constants import DEFAULT_NAMESPACE
from typing import List
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from streamz import Stream
import ai_flow as af
from ai_flow import ModelMeta
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage, ModelVersionEventType
from notification_service.base_notification import EventWatcher, BaseEvent
from python_ai_flow import FunctionContext, Executor, ExampleExecutor
from ai_flow.common.path_util import get_file_dir


class ExampleTrainThread(threading.Thread):

    def __init__(self, stream_uri):
        super().__init__()
        self.stream_uri = stream_uri
        self.stream = Stream()

    def run(self) -> None:
        for i in range(0, 3):
            f = np.load(self.stream_uri)
            x_train, y_train = f['x_train'], f['y_train']
            f.close()
            self.stream.emit((x_train, y_train))
            time.sleep(60)


class ExampleTrain(ExampleExecutor):

    def __init__(self):
        super().__init__()
        self.thread = None

    def setup(self, function_context: FunctionContext):
        self.thread = ExampleTrainThread(function_context.node_spec.example_meta.stream_uri)
        self.thread.start()

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        return [self.thread.stream]


class TransformTrain(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        def transform(df):
            x_train, y_train = df[0], df[1]
            random_state = check_random_state(0)
            permutation = random_state.permutation(x_train.shape[0])
            x_train, y_train = x_train[permutation], y_train[permutation]
            x_train = x_train.reshape((x_train.shape[0], -1))
            return StandardScaler().fit_transform(x_train), y_train

        return [input_list[0].map(transform)]


class TrainModel(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        def train(df):
            print("Do train")
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

            # if af.get_model_by_name(model.name) is None:
            af.register_model_version(model=model, model_path=model_path, current_stage=ModelVersionStage.GENERATED)
            print("Register done")
            # else:
            #     print("update model")
            #     af.update_model_version(model_name=model.name, model_version=model_version, model_path=model_path, current_stage=ModelVersionStage.GENERATED)
            return df

        def sink(df):
            pass

        input_list[0].map(train).sink(sink)
        return []


class ExampleEvaluate(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        f = np.load(function_context.node_spec.example_meta.stream_uri)
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        return [[x_test, y_test]]


class TransformEvaluate(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        x_test, y_test = input_list[0][0], input_list[0][1]
        random_state = check_random_state(0)
        permutation = random_state.permutation(x_test.shape[0])
        x_test, y_test = x_test[permutation], y_test[permutation]
        x_test = x_test.reshape((x_test.shape[0], -1))
        return [[StandardScaler().fit_transform(x_test), y_test]]


class EvaluateModel(Executor):

    def __init__(self):
        super().__init__()
        self.model_path = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        print("### {} setup {}".format(self.__class__.__name__, function_context.node_spec.model.name))

        class EvaluateWatcher(EventWatcher):
            def __init__(self):
                self.queue: queue.Queue = queue.Queue(1)

            def process(self, events: List[BaseEvent]):
                print("events: " + str(events))
                self.queue.put(events[0])

            def get_result(self) -> object:
                return self.queue.get()
        watcher = EvaluateWatcher()
        af.start_listen_event(key=function_context.node_spec.model.name,
                              watcher=watcher,
                              namespace=DEFAULT_NAMESPACE,
                              event_type=ModelVersionEventType.MODEL_GENERATED)
        event = watcher.get_result()
        self.model_path = json.loads(event.value).get('_model_path')
        self.model_version = json.loads(event.value).get('_model_version')

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        x_evaluate, y_evaluate = input_list[0][0], input_list[0][1]
        clf = load(self.model_path)
        scores = cross_val_score(clf, x_evaluate, y_evaluate, cv=5)
        evaluate_artifact = af.get_artifact_by_name('evaluate_artifact').stream_uri
        with open(evaluate_artifact, 'a') as f:
            f.write('model version[{}] scores: {}\n'.format(self.model_version, scores))
        return []


class ExampleValidate(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        f = np.load(function_context.node_spec.example_meta.stream_uri)
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        return [[x_test, y_test]]


class TransformValidate(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        x_test, y_test = input_list[0][0], input_list[0][1]
        random_state = check_random_state(0)
        permutation = random_state.permutation(x_test.shape[0])
        x_test, y_test = x_test[permutation], y_test[permutation]
        x_test = x_test.reshape((x_test.shape[0], -1))
        return [[StandardScaler().fit_transform(x_test), y_test]]


class ValidateModel(Executor):

    def __init__(self):
        super().__init__()
        self.model_path = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        class ValidateWatcher(EventWatcher):

            def process(self, notifications):
                for notification in notifications:
                    print(self.__class__.__name__)
                    print(notification)
        notifications = af.start_listen_event(key=function_context.node_spec.model.name, watcher=ValidateWatcher())
        self.model_path = json.loads(notifications[len(notifications) - 1].value).get('_model_path')
        self.model_version = json.loads(notifications[len(notifications) - 1].value).get('_model_version')

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        model_meta: ModelMeta = function_context.node_spec.model
        serving_model_version = af.get_deployed_model_version(model_name=model_meta.name)
        if serving_model_version is None:
            af.update_model_version(model_name=model_meta.name,
                                    model_version=self.model_version,
                                    current_stage=ModelVersionStage.DEPLOYED)
            af.send_event(key='serving_model_version',
                                   value=self.model_version)
        else:
            x_validate, y_validate = input_list[0][0], input_list[0][1]
            clf = load(self.model_path)
            scores = cross_validate(clf, x_validate, y_validate, scoring='precision_macro', cv=5, return_estimator=True)
            serving_clf = load(serving_model_version.model_path)
            serving_scores = cross_validate(serving_clf, x_validate, y_validate, scoring='precision_macro',
                                            return_estimator=True)
            stream_uri = af.get_artifact_by_name('validate_artifact').batch_uri
            with open(stream_uri, 'a') as f:
                f.write('serving model version[{}] scores: {}\n'.format(serving_model_version.version, serving_scores))
                f.write('generated model version[{}] scores: {}\n'.format(self.model_version, scores))
            if scores['test_score'].mean() > serving_scores['test_score'].mean():
                af.update_model_version(model_name=model_meta.name,
                                        model_version=serving_model_version.version,
                                        current_stage=ModelVersionStage.VALIDATED)
                af.update_model_version(model_name=model_meta.name,
                                        model_version=self.model_version,
                                        current_stage=ModelVersionStage.DEPLOYED)
                af.send_event(key='serving_model_version',
                                       value=self.model_version)
        return []


class ExamplePredictThread(threading.Thread):

    def __init__(self, stream_uri):
        super().__init__()
        self.stream_uri = stream_uri
        self.stream = Stream()

    def run(self) -> None:
        for i in range(0, 20):
            f = np.load(self.stream_uri)
            x_test = f['x_test']
            f.close()
            self.stream.emit(x_test)
            time.sleep(2)


class ExamplePredict(ExampleExecutor):

    def __init__(self):
        super().__init__()
        self.thread = None

    def setup(self, function_context: FunctionContext):
        self.thread = ExamplePredictThread(function_context.node_spec.example_meta.stream_uri)
        self.thread.start()

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        return [self.thread.stream]


class TransformPredict(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        def transform(df):
            x_test = df
            random_state = check_random_state(0)
            permutation = random_state.permutation(x_test.shape[0])
            x_test = x_test[permutation]
            x_test = x_test.reshape((x_test.shape[0], -1))
            return StandardScaler().fit_transform(x_test)

        return [input_list[0].map(transform)]


class PredictWatcher(EventWatcher):

    def __init__(self):
        super().__init__()
        self.model_version = None

    def process(self, notifications):
        for notification in notifications:
            self.model_version = notification.value


class PredictModel(Executor):

    def __init__(self):
        super().__init__()
        self.watcher = PredictWatcher()

    def setup(self, function_context: FunctionContext):
        af.start_listen_event(key='serving_model_version',
                              watcher=self.watcher)

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        while self.watcher.model_version is None:
            pass

        def predict(df):
            x_test = df
            model_path = af.get_model_version_by_version(version=self.watcher.model_version,
                                                           model_id=function_context.node_spec.model.uuid).model_path
            clf = load(model_path)
            return self.watcher.model_version, clf.predict(x_test)

        return [input_list[0].map(predict)]


class WriteExample(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("### {}".format(self.__class__.__name__))
        def write_example(df):
            with open(function_context.node_spec.example_meta.stream_uri, 'a') as f:
                f.write('model version[{}] predict: {}\n'.format(df[0], df[1]))
            return df

        def sink(df):
            pass

        input_list[0].map(write_example).sink(sink)
        return []
