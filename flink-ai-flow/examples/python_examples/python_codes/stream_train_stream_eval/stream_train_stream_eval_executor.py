import json
import os
import threading
import time
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
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage
from notification_service.base_notification import EventWatcher
from python_ai_flow import FunctionContext, Executor, ExampleExecutor
from ai_flow.common.path_util import get_file_dir
from ai_flow.model_center.entity.model_version_stage import ModelVersionEventType


class ExampleTrainThread(threading.Thread):

    def __init__(self, stream_uri):
        super().__init__()
        self.stream_uri = stream_uri
        self.stream = Stream()

    def run(self) -> None:
        for i in range(0, 20):
            f = np.load(self.stream_uri)
            x_train, y_train = f['x_train'], f['y_train']
            f.close()
            self.stream.emit((x_train, y_train))
            time.sleep(2)


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
            model_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            model_path = model_path + '/' + model_timestamp
            dump(clf, model_path)
            model = function_context.node_spec.output_model
            print(model.name)
            print(model_timestamp)

            af.register_model_version(model=model, model_path=model_path)
            print("Register done")
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


class EvaluateWatcher(EventWatcher):
    def __init__(self):
        super().__init__()
        self.events = []

    def process(self, notifications):
        for notification in notifications:
            print(self.__class__.__name__)
            print(notification)
            self.events.append(notification)


class EvaluateModel(Executor):

    def __init__(self):
        super().__init__()
        self.model_path = None
        self.model_version = None
        self.watcher = EvaluateWatcher()

    def setup(self, function_context: FunctionContext):
        evaluate_artifact = af.get_artifact_by_name('evaluate_artifact').stream_uri
        with open(evaluate_artifact, 'a') as f:
            f.write('model version[{}] scores: {}\n'.format("sss", "ddd"))
        print("### {} setup {}".format(self.__class__.__name__, function_context.node_spec.model.name))
        af.start_listen_event(key=function_context.node_spec.model.name, watcher=self.watcher)
        self.model_path = json.loads(self.watcher.events[len(self.watcher.events) - 1].value).get('_model_path')
        self.model_version = json.loads(self.watcher.events[len(self.watcher.events) - 1].value).get('_model_version')

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        evaluate_artifact = af.get_artifact_by_name('evaluate_artifact').stream_uri
        with open(evaluate_artifact, 'a') as f:
            f.write('model version[{}] scores: {}\n'.format("sss", "ddd"))
        print("### {}".format(self.__class__.__name__))
        x_evaluate, y_evaluate = input_list[0][0], input_list[0][1]
        clf = load(self.model_path)
        scores = cross_val_score(clf, x_evaluate, y_evaluate, cv=5)

        print(evaluate_artifact)
        with open(evaluate_artifact, 'a') as f:
            f.write('model version[{}] scores: {}\n'.format(self.model_version, scores))
        return []

