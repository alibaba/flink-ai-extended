import json
import os
import time

import ai_flow as af
import numpy as np
from ai_flow import ModelMeta
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage
from joblib import dump, load
from typing import List
from python_ai_flow import FunctionContext, Executor, ExampleExecutor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split

EXAMPLE_COLUMNS = ['facidity', 'vacidity', 'cacid', 'rsugar', 'tdioxide', 'density', 'pH', 'sulphates', 'alcohol',
                   'quality']


def read_example_properties(batch_uri):
    return {'connector.type': 'filesystem',
            'connector.path': batch_uri,
            'connector.property-version': '1',
            'format.type': 'csv',
            'format.property-version': '1',
            'format.ignore-first-line': True,
            'format.fields.0.type': 'DOUBLE',
            'format.fields.1.type': 'DOUBLE',
            'format.fields.2.type': 'DOUBLE',
            'format.fields.3.type': 'DOUBLE',
            'format.fields.4.type': 'DOUBLE',
            'format.fields.5.type': 'DOUBLE',
            'format.fields.6.type': 'DOUBLE',
            'format.fields.7.type': 'DOUBLE',
            'format.fields.8.type': 'DOUBLE',
            'format.fields.9.type': 'DOUBLE',
            'format.fields.10.type': 'DOUBLE',
            'format.fields.11.type': 'DOUBLE',
            'format.fields.0.name': 'facidity',
            'format.fields.1.name': 'vacidity',
            'format.fields.2.name': 'cacid',
            'format.fields.3.name': 'rsugar',
            'format.fields.4.name': 'chlorides',
            'format.fields.5.name': 'fdioxide',
            'format.fields.6.name': 'tdioxide',
            'format.fields.7.name': 'density',
            'format.fields.8.name': 'pH',
            'format.fields.9.name': 'sulphates',
            'format.fields.10.name': 'alcohol',
            'format.fields.11.name': 'quality',
            'schema.0.type': 'DOUBLE',
            'schema.1.type': 'DOUBLE',
            'schema.2.type': 'DOUBLE',
            'schema.3.type': 'DOUBLE',
            'schema.4.type': 'DOUBLE',
            'schema.5.type': 'DOUBLE',
            'schema.6.type': 'DOUBLE',
            'schema.7.type': 'DOUBLE',
            'schema.8.type': 'DOUBLE',
            'schema.9.type': 'DOUBLE',
            'schema.10.type': 'DOUBLE',
            'schema.11.type': 'DOUBLE',
            'schema.0.name': 'facidity',
            'schema.1.name': 'vacidity',
            'schema.2.name': 'cacid',
            'schema.3.name': 'rsugar',
            'schema.4.name': 'chlorides',
            'schema.5.name': 'fdioxide',
            'schema.6.name': 'tdioxide',
            'schema.7.name': 'density',
            'schema.8.name': 'pH',
            'schema.9.name': 'sulphates',
            'schema.10.name': 'alcohol',
            'schema.11.name': 'quality'}


def write_example_properties(batch_uri):
    return {'connector.type': 'filesystem',
            'connector.path': batch_uri,
            'connector.property-version': '1',
            'format.type': 'csv',
            'format.property-version': '1',
            'format.fields.0.type': 'DOUBLE',
            'format.fields.1.type': 'DOUBLE',
            'format.fields.2.type': 'DOUBLE',
            'format.fields.3.type': 'DOUBLE',
            'format.fields.4.type': 'DOUBLE',
            'format.fields.5.type': 'DOUBLE',
            'format.fields.6.type': 'DOUBLE',
            'format.fields.7.type': 'DOUBLE',
            'format.fields.8.type': 'DOUBLE',
            'format.fields.9.type': 'DOUBLE',
            'format.fields.0.name': 'facidity',
            'format.fields.1.name': 'vacidity',
            'format.fields.2.name': 'cacid',
            'format.fields.3.name': 'rsugar',
            'format.fields.4.name': 'tdioxide',
            'format.fields.5.name': 'density',
            'format.fields.6.name': 'pH',
            'format.fields.7.name': 'sulphates',
            'format.fields.8.name': 'alcohol',
            'format.fields.9.name': 'quality',
            'schema.0.type': 'DOUBLE',
            'schema.1.type': 'DOUBLE',
            'schema.2.type': 'DOUBLE',
            'schema.3.type': 'DOUBLE',
            'schema.4.type': 'DOUBLE',
            'schema.5.type': 'DOUBLE',
            'schema.6.type': 'DOUBLE',
            'schema.7.type': 'DOUBLE',
            'schema.8.type': 'DOUBLE',
            'schema.9.type': 'DOUBLE',
            'schema.0.name': 'facidity',
            'schema.1.name': 'vacidity',
            'schema.2.name': 'cacid',
            'schema.3.name': 'rsugar',
            'schema.4.name': 'tdioxide',
            'schema.5.name': 'density',
            'schema.6.name': 'pH',
            'schema.7.name': 'sulphates',
            'schema.8.name': 'alcohol',
            'schema.9.name': 'quality'}


class TrainModel(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        lr = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        train, test = train_test_split(input_list[0])
        x_train, y_train = train.drop(train.columns[-1], axis=1), train[train.columns[-1]]
        lr.fit(x_train, y_train)
        model_path = 'saved_model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_version = time.strftime('%Y%m%d%H%M%S', time.localtime())
        model_path = model_path + '/' + model_version
        dump(lr, model_path)
        af.register_model_version(model=function_context.node_spec.output_model, model_path=model_path)
        return []


class EvaluateModel(Executor):

    def __init__(self):
        super().__init__()
        self.mode = None
        self.model_path = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        # notifications = af.start_listen_notification(listener_name='evaluate_listener',
        #                                              key=function_context.node_spec.model.name)
        # print('=====================================')
        # print(notifications)
        # print('=====================================')
        self.model = af.get_latest_generated_model_version(function_context.node_spec.model.name)
        self.model_path = self.model.model_path
        self.model_version = self.model.version

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        train, test = train_test_split(input_list[0])
        x_evaluate, y_evaluate = test.drop(test.columns[-1], axis=1), test[test.columns[-1]]
        clf = load(self.model_path)
        scores = cross_val_score(clf, x_evaluate, y_evaluate, cv=5)
        evaluate_artifact = af.get_artifact_by_name('evaluate_artifact').batch_uri
        with open(evaluate_artifact, 'a') as f:
            f.write('model version[{}] scores: {}\n'.format(self.model_version, scores))
        return []


class ValidateModel(Executor):

    def __init__(self):
        super().__init__()
        self.mode = None
        self.model_path = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        # notifications = af.start_listen_notification(listener_name='validate_listener',
        #                                              key=function_context.node_spec.model.name)
        self.model = af.get_latest_generated_model_version(function_context.node_spec.model.name)
        self.model_path = self.model.model_path
        self.model_version = self.model.version

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        new_model_version = self.model_version
        model_meta: ModelMeta = function_context.node_spec.model
        deployed_model_version = af.get_deployed_model_version(model_name=model_meta.name)
        if deployed_model_version is None:
            af.update_model_version(model_name=model_meta.name, model_version=new_model_version,
                                    current_stage=ModelVersionStage.DEPLOYED)
        else:
            train, test = train_test_split(input_list[0])
            x_validate, y_validate = test.drop(test.columns[-1], axis=1), test[test.columns[-1]]
            clf = load(self.model_path)
            scoring = ['precision_macro', 'recall_macro']
            scores = cross_validate(clf, x_validate, y_validate, scoring=scoring)
            deployed_clf = load(deployed_model_version.model_path)
            deployed_scores = cross_validate(deployed_clf, x_validate, y_validate, scoring=scoring)
            batch_uri = af.get_artifact_by_name('validate_artifact').batch_uri
            with open(batch_uri, 'a') as f:
                f.write('deployed model version[{}] scores: {}\n'.format(deployed_model_version.version, deployed_scores))
                f.write('generated model version[{}] scores: {}\n'.format(self.model_version, scores))
            if scores.mean() > deployed_scores.mean():
                af.update_model_version(model_name=model_meta.name,
                                        model_version=deployed_model_version.version,
                                        current_stage=ModelVersionStage.VALIDATED)
                af.update_model_version(model_name=model_meta.name,
                                        model_version=new_model_version,
                                        current_stage=ModelVersionStage.DEPLOYED)
        return []


class PredictModel(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        model_version = af.get_deployed_model_version(function_context.node_spec.model.name)
        clf = load(model_version.model_path)
        train, test = train_test_split(input_list[0])
        return [clf.predict(test.drop(test.columns[-1], axis=1))]


class WriteExample(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        np.savetxt(function_context.node_spec.example_meta.batch_uri, input_list[0])
        return []
