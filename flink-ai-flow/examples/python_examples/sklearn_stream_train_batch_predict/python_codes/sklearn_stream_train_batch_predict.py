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
import ai_flow as af

from ai_flow import ExampleSupportType, PythonObjectExecutor, ModelType
from ai_flow.common.scheduler_type import SchedulerType
from ai_flow.model_center.entity.model_version_stage import ModelVersionEventType
from ai_flow.common.path_util import get_file_dir
from stream_train_batch_predict_executor import TrainExampleReader, TrainExampleTransformer, ModelTrainer, \
    ValidateExampleReader, ValidateTransformer, ModelValidator, PredictExampleReader, PredictTransformer, \
    ModelPredictor, ExampleWriter, ModelPusher

EXAMPLE_URI = os.path.abspath(os.path.join(__file__, "../../../..")) + '/example_data/mnist_{}.npz'


def run_project(project_root_path):
    af.set_project_config_file(project_root_path + "/project.yaml")
    project_name = af.project_config().get_project_name()
    artifact_prefix = project_name + "."

    validate_trigger = af.external_trigger(name='validate')
    push_trigger = af.external_trigger(name='push')

    with af.global_config_file(project_root_path + '/resources/workflow_config.yaml'):
        with af.config('train_job'):
            train_example = af.register_example(name=artifact_prefix + 'train_example',
                                                support_type=ExampleSupportType.EXAMPLE_STREAM,
                                                stream_uri=EXAMPLE_URI.format('train'))
            train_read_example = af.read_example(example_info=train_example,
                                                 executor=PythonObjectExecutor(python_object=TrainExampleReader()))
            train_transform = af.transform(input_data_list=[train_read_example],
                                           executor=PythonObjectExecutor(python_object=TrainExampleTransformer()))
            train_model = af.register_model(model_name=artifact_prefix + 'logistic-regression',
                                            model_type=ModelType.SAVED_MODEL,
                                            model_desc='logistic regression model')
            train_channel = af.train(input_data_list=[train_transform],
                                     executor=PythonObjectExecutor(python_object=ModelTrainer()),
                                     model_info=train_model)
        with af.config('validate_job'):
            validate_example = af.register_example(name=artifact_prefix + 'validate_example',
                                                   support_type=ExampleSupportType.EXAMPLE_STREAM,
                                                   stream_uri=EXAMPLE_URI.format('evaluate'),
                                                   data_format='npz')
            validate_read_example = af.read_example(example_info=validate_example,
                                                    executor=PythonObjectExecutor(
                                                        python_object=ValidateExampleReader()))
            validate_transform = af.transform(input_data_list=[validate_read_example],
                                              executor=PythonObjectExecutor(python_object=ValidateTransformer()))
            validate_artifact_name = artifact_prefix + 'validate_artifact'
            validate_artifact = af.register_artifact(name=validate_artifact_name,
                                                     stream_uri=get_file_dir(__file__) + '/validate_model')
            validate_channel = af.model_validate(input_data_list=[validate_transform],
                                                 model_info=train_model,
                                                 executor=PythonObjectExecutor(
                                                     python_object=ModelValidator(validate_artifact_name)),
                                                 )
        with af.config('push_job'):
            # Push model to serving
            # Register metadata of pushed model
            push_model_artifact_name = artifact_prefix + 'push_model_artifact'
            push_model_artifact = af.register_artifact(name=push_model_artifact_name,
                                                       stream_uri=get_file_dir(__file__) + '/pushed_model')
            push_channel = af.push_model(model_info=train_model,
                                         executor=PythonObjectExecutor(
                                            python_object=ModelPusher(push_model_artifact_name)))

        with af.config('predict_job'):
            predict_example = af.register_example(name=artifact_prefix + 'predict_example',
                                                  support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                  batch_uri=EXAMPLE_URI.format('predict'))
            predict_read_example = af.read_example(example_info=predict_example,
                                                   executor=PythonObjectExecutor(python_object=PredictExampleReader()))
            predict_transform = af.transform(input_data_list=[predict_read_example],
                                             executor=PythonObjectExecutor(python_object=PredictTransformer()))
            predict_channel = af.predict(input_data_list=[predict_transform],
                                         model_info=train_model,
                                         executor=PythonObjectExecutor(python_object=ModelPredictor()))

            write_example = af.register_example(name=artifact_prefix + 'export_example',
                                                support_type=ExampleSupportType.EXAMPLE_BATCH,
                                                batch_uri=get_file_dir(__file__) + '/predict_result')
            af.write_example(input_data=predict_channel,
                             example_info=write_example,
                             executor=PythonObjectExecutor(python_object=ExampleWriter()))

        af.model_version_control_dependency(src=validate_channel,
                                            model_version_event_type=ModelVersionEventType.MODEL_GENERATED,
                                            dependency=validate_trigger, model_name=train_model.name)
        af.model_version_control_dependency(src=push_channel,
                                            model_version_event_type=ModelVersionEventType.MODEL_VALIDATED,
                                            dependency=push_trigger, model_name=train_model.name)

    # Run workflow
    transform_dag = project_name
    af.deploy_to_airflow(project_root_path, dag_id=transform_dag)
    af.run(project_path=project_root_path,
           dag_id=transform_dag,
           scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    project_path = os.path.dirname(get_file_dir(__file__))
    run_project(project_path)
