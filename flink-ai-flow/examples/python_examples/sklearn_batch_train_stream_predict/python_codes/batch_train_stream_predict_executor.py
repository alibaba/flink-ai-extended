#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
import os
import shutil
import threading
import time
from typing import List
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import ai_flow as af
from streamz import Stream
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage
from python_ai_flow import FunctionContext, Executor, ExampleExecutor
from ai_flow.common.path_util import get_file_dir
from notification_service.base_notification import EventWatcher


def preprocess_data(x_data, y_data=None):
    random_state = check_random_state(0)
    permutation = random_state.permutation(x_data.shape[0])
    if y_data is None:
        return x_data[permutation]
    else:
        return x_data[permutation], y_data[permutation]


class ExampleReader(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        with np.load(function_context.node_spec.example_meta.batch_uri) as f:
            x_train, y_train = f['x_train'], f['y_train']
            f.close()
        return [[x_train, y_train]]


class ExampleTransformer(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        x_train, y_train = preprocess_data(input_list[0][0], input_list[0][1])
        x_train = x_train.reshape((x_train.shape[0], -1))
        res = [[StandardScaler().fit_transform(x_train), y_train]]
        return res


class ModelTrainer(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html
        clf = LogisticRegression(C=50. / 5000, penalty='l1', solver='saga', tol=1)
        x_train, y_train = input_list[0][0], input_list[0][1]
        clf.fit(x_train, y_train)
        model_path = get_file_dir(__file__) + '/saved_model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        model_path = model_path + '/' + model_timestamp
        dump(clf, model_path)
        af.register_model_version(model=function_context.node_spec.output_model, model_path=model_path)
        print("Done for {}".format(self.__class__.__name__))
        return []


class ValidateExampleReader(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        with np.load(function_context.node_spec.example_meta.batch_uri) as f:
            x_test, y_test = f['x_test'], f['y_test']
        return [[x_test, y_test]]


class ValidateTransformer(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        x_test, y_test = preprocess_data(input_list[0][0], input_list[0][1])
        x_test = x_test.reshape((x_test.shape[0], -1))
        return [[StandardScaler().fit_transform(x_test), y_test]]


class ModelValidator(Executor):

    def __init__(self, artifact_name):
        super().__init__()
        self.artifact_name = artifact_name
        self.model_name = None
        self.model_path = None
        self.model_version = None
        self.model_meta = None

    def setup(self, function_context: FunctionContext):
        self.model_name = function_context.node_spec.model.name
        self.model_meta = af.get_latest_generated_model_version(self.model_name)
        self.model_path = self.model_meta.model_path
        self.model_version = self.model_meta.version

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        deployed_model_version = af.get_deployed_model_version(model_name=self.model_name)
        x_validate, y_validate = input_list[0][0], input_list[0][1]
        clf = load(self.model_path)
        scores = cross_val_score(clf, x_validate, y_validate, scoring='precision_macro')
        batch_uri = af.get_artifact_by_name(self.artifact_name).batch_uri
        if deployed_model_version is None:
            with open(batch_uri, 'a') as f:
                f.write('generated model version[{}] scores: {}\n'.format(self.model_version, np.mean(scores)))
            af.update_model_version(model_name=self.model_name,
                                    model_version=self.model_version,
                                    current_stage=ModelVersionStage.VALIDATED)
        else:
            deployed_clf = load(deployed_model_version.model_path)
            deployed_scores = cross_val_score(deployed_clf, x_validate, y_validate, scoring='precision_macro')
            f = open(batch_uri, 'a')
            f.write('current model version[{}] scores: {}\n'.format(deployed_model_version.version,
                                                                    np.mean(deployed_scores)))
            f.write('new generated model version[{}] scores: {}\n'.format(self.model_version, np.mean(scores)))
            if np.mean(scores) > np.mean(deployed_scores):
                # Make latest generated model to be validated
                af.update_model_version(model_name=self.model_name,
                                        model_version=self.model_version,
                                        current_stage=ModelVersionStage.VALIDATED)
                f.write('new generated model version[{}] pass validation.\n'.format(self.model_version))
            else:
                f.write('new generated model version[{}] fail validation.\n'.format(self.model_version))
            f.close()

        return []


class ModelPusher(Executor):
    def __init__(self, artifact):
        super().__init__()
        self.artifact = artifact

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        model_name = function_context.node_spec.model.name
        validated_model = af.get_latest_validated_model_version(model_name)
        # Deprecate deployed model
        deployed_model_version = af.get_deployed_model_version(model_name)
        if deployed_model_version is not None:
            af.update_model_version(model_name=model_name,
                                    model_version=deployed_model_version.version,
                                    current_stage=ModelVersionStage.DEPRECATED)
        af.update_model_version(model_name=model_name,
                                model_version=validated_model.version,
                                current_stage=ModelVersionStage.DEPLOYED)

        af.send_event(key='START_PREDICTION', value=validated_model.version)
        print(validated_model.version)

        # Copy deployed model to deploy_model_dir

        deployed_model_dir = af.get_artifact_by_name(self.artifact).batch_uri
        if not os.path.exists(deployed_model_dir):
            os.makedirs(deployed_model_dir)
        for file in os.listdir(deployed_model_dir):
            file_path = os.path.join(deployed_model_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, True)
        deployed_model_version = af.get_deployed_model_version(model_name)
        shutil.copy(deployed_model_version.model_path, deployed_model_dir)
        return []


class ExamplePredictThread(threading.Thread):

    def __init__(self, stream_uri):
        super().__init__()
        self.stream_uri = stream_uri
        self.stream = Stream()

    def run(self) -> None:
        for i in range(0, 8):
            with np.load(self.stream_uri) as f:
                x_test = f['x_test']
            self.stream.emit(x_test)
            print("### {} {}".format(self.__class__.__name__, "generate data flow"))
            time.sleep(50)


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


class PredictWatcher(EventWatcher):

    def __init__(self):
        super().__init__()
        self.model_version = None

    def process(self, notifications):
        for notification in notifications:
            self.model_version = notification.value


class ModelPredictor(Executor):

    def __init__(self):
        super().__init__()
        self.model_name = None
        self.model_version = None
        self.watcher = PredictWatcher()

    def setup(self, function_context: FunctionContext):
        # In this class, we show the usage of start_listen_event method which make it possible to send various events.
        # Users can also refer `stream train stream predict` example to directly use provided API to get model version.
        af.start_listen_event(key='START_PREDICTION', watcher=self.watcher)
        self.model_name = function_context.node_spec.model.name
        print("### {} setup done for {}".format(self.__class__.__name__, function_context.node_spec.model.name))

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        while self.watcher.model_version is None:
            time.sleep(2)
        print("### {} ".format(self.watcher.model_version))

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
