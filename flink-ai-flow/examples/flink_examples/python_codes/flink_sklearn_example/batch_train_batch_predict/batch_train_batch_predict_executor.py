import json
import os
import time

import ai_flow as af
import numpy as np
from ai_flow import ModelMeta
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage
from joblib import dump, load
from python_ai_flow import FunctionContext, List, Executor, ExampleExecutor
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
        model_source = 'saved_model'
        if not os.path.exists(model_source):
            os.makedirs(model_source)
        model_version = time.strftime('%Y%m%d%H%M%S', time.localtime())
        model_source = model_source + '/' + model_version
        dump(lr, model_source)
        af.register_model_version(version=model_version,
                                  model_id=function_context.node_spec.output_model.uuid,
                                  workflow_execution_id=function_context.job_context.workflow_execution_id,
                                  model_source=model_source)
        return []


class EvaluateModel(Executor):

    def __init__(self):
        super().__init__()
        self.model_source = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        notifications = af.start_listen_notification(listener_name='evaluate_listener',
                                                     key=function_context.node_spec.model.name)
        print('=====================================')
        print(notifications)
        print('=====================================')
        notification = notifications[0].value
        self.model_source = json.loads(notification).get('_model_source')
        self.model_version = json.loads(notification).get('_model_version')

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        train, test = train_test_split(input_list[0])
        x_evaluate, y_evaluate = test.drop(test.columns[-1], axis=1), test[test.columns[-1]]
        clf = load(self.model_source)
        scores = cross_val_score(clf, x_evaluate, y_evaluate, cv=5)
        with open(function_context.node_spec.output_result.batch_uri, 'a') as f:
            f.write('model version[{}] scores: {}\n'.format(self.model_version, scores))
        return []


class ValidateModel(Executor):

    def __init__(self):
        super().__init__()
        self.model_source = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        notifications = af.start_listen_notification(listener_name='validate_listener',
                                                     key=function_context.node_spec.model.name)
        self.model_source = json.loads(notifications[0].value).get('_model_source')
        self.model_version = json.loads(notifications[0].value).get('_model_version')

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        new_model_version = self.model_version
        model_meta: ModelMeta = function_context.node_spec.model
        serving_model_version = af.get_serving_model_version(model_name=model_meta.name)
        if serving_model_version is None:
            af.update_model_version(model_name=model_meta.name, model_version=new_model_version,
                                    current_stage=ModelVersionStage.DEPLOYED)
        else:
            train, test = train_test_split(input_list[0])
            x_validate, y_validate = test.drop(test.columns[-1], axis=1), test[test.columns[-1]]
            clf = load(self.model_source)
            scoring = ['precision_macro', 'recall_macro']
            scores = cross_validate(clf, x_validate, y_validate, scoring=scoring)
            serving_clf = load(serving_model_version.model_source)
            serving_scores = cross_validate(serving_clf, x_validate, y_validate, scoring=scoring)
            with open(function_context.node_spec.output_result.batch_uri, 'a') as f:
                f.write('serving model version[{}] scores: {}\n'.format(serving_model_version.version, serving_scores))
                f.write('generated model version[{}] scores: {}\n'.format(self.model_version, scores))
            if scores.mean() > serving_scores.mean():
                af.update_model_version(model_name=model_meta.name,
                                        model_version=serving_model_version.version,
                                        current_stage=ModelVersionStage.VALIDATED)
                af.update_model_version(model_name=model_meta.name,
                                        model_version=new_model_version,
                                        current_stage=ModelVersionStage.DEPLOYED)
        return []


class PredictModel(Executor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        model_version = af.get_serving_model_version(function_context.node_spec.model.name)
        clf = load(model_version.model_source)
        train, test = train_test_split(input_list[0])
        return [clf.predict(test.drop(test.columns[-1], axis=1))]


class WriteExample(ExampleExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        np.savetxt(function_context.node_spec.example_meta.batch_uri, input_list[0])
        return []
