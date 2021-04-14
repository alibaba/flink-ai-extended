from ai_flow import ExecutionMode, ExampleSupportType, ExecuteArgs, BaseJobConfig, PythonObjectExecutor, ModelType, \
    ArtifactMeta
from ai_flow.common.scheduler_type import SchedulerType
from ai_flow.common.args import Args
from ai_flow.common.path_util import get_file_dir, get_parent_dir

from ai_flow.model_center.entity.model_version_stage import ModelVersionEventType
import example_util
from flink_ai_flow import FlinkPythonExecutor
from flink_ai_flow import LocalFlinkJobConfig
from stream_train_stream_predict_pyflink_executor import *


def get_project_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXAMPLE_DATASET = get_project_path() + '/dataset/{}.csv'


def run_job():
    project_root_path = example_util.get_root_path()
    af.set_project_config_file(project_root_path + '/project.yaml')
    python_job_config = BaseJobConfig()
    python_job_config.platform = 'local'
    python_job_config.engine = 'python'
    python_job_config.exec_mode = ExecutionMode.STREAM
    evaluate_trigger = af.external_trigger('evaluate_trigger')
    validate_trigger = af.external_trigger('validate_trigger')
    with af.config(python_job_config):
        train_example = af.register_example(name='train_example9',
                                            support_type=ExampleSupportType.EXAMPLE_STREAM,
                                            stream_uri=EXAMPLE_DATASET.format('iris_train'),
                                            data_format='csv')
        train_model = af.register_model(model_name='iris_model',
                                        model_type=ModelType.SAVED_MODEL,
                                        model_desc='hello, iris model')
        train_read_example_channel = af.read_example(example_info=train_example,
                                                     executor=PythonObjectExecutor(
                                                         python_object=ReadTrainExample.LoadTrainExample()))
        train_channel = af.train(input_data_list=[train_read_example_channel],
                                 executor=PythonObjectExecutor(python_object=TrainModel()),
                                 model_info=train_model)

        evaluate_example = af.register_example(name='evaluate_example9', support_type=ExampleSupportType.EXAMPLE_STREAM,
                                               data_format='csv',
                                               batch_uri=EXAMPLE_DATASET.format('iris_test'),
                                               stream_uri=EXAMPLE_DATASET.format('iris_test'))
        evaluate_example_channel = af.read_example(example_info=evaluate_example, exec_args=ExecuteArgs(
            batch_properties=Args(header=0, names=EXAMPLE_COLUMNS),
            stream_properties=Args(header=0, names=EXAMPLE_COLUMNS)), executor=PythonObjectExecutor(python_object=EvaluateExampleReader()))
        evaluate_result = get_file_dir(__file__) + '/evaluate_result'
        if os.path.exists(evaluate_result):
            os.remove(evaluate_result)
        evaluate_artifact: ArtifactMeta = af.register_artifact(name='evaluate_artifact9',
                                                               batch_uri=evaluate_result,
                                                               stream_uri=evaluate_result)
        evaluate_channel = af.evaluate(input_data_list=[evaluate_example_channel], model_info=train_model,
                                       executor=PythonObjectExecutor(python_object=EvaluateModel()))

    """configure local mode of job config"""
    job_config_1 = LocalFlinkJobConfig()
    job_config_1.local_mode = 'python'
    """configure table environment create function of job config"""
    job_config_1.set_table_env_create_func(StreamTableEnvCreator())
    predict_trigger = af.external_trigger("predict")
    with af.config(job_config_1):
        predict_example = af.register_example(name='predict_example9',
                                              support_type=ExampleSupportType.EXAMPLE_BATCH,
                                              batch_uri=EXAMPLE_DATASET.format('iris_test'),
                                              stream_uri=EXAMPLE_DATASET.format('iris_test'),
                                              data_format='csv')
        predict_read_example = af.read_example(example_info=predict_example,
                                               executor=FlinkPythonExecutor(python_object=Source()))
        predict_channel = af.predict(input_data_list=[predict_read_example],
                                     model_info=train_model,
                                     executor=FlinkPythonExecutor(python_object=Transformer()))

        write_example = af.register_example(name='write_example9',
                                            support_type=ExampleSupportType.EXAMPLE_BATCH,
                                            batch_uri=get_file_dir(
                                                __file__) + '/predict_model.csv',
                                            stream_uri=get_file_dir(
                                                __file__) + '/predict_model.csv',
                                            data_format='csv')
        af.write_example(input_data=predict_channel,
                         example_info=write_example,
                         executor=PythonObjectExecutor(python_object=Sink()))

    # af.model_version_control_dependency(evaluate_channel, train_model.name, evaluate_trigger)
    af.model_version_control_dependency(src=evaluate_channel,
                                        model_version_event_type=ModelVersionEventType.MODEL_GENERATED,
                                        dependency=evaluate_trigger, model_name=train_model.name)


    # af.model_version_control_dependency(validate_channel, train_model.name, validate_trigger)
    # af.model_version_control_dependency(src=validate_channel,
    #                                     model_version_event_type=ModelVersionEventType.MODEL_GENERATED,
    #                                     dependency=validate_trigger, model_name=train_model.name)

    af.model_version_control_dependency(src=predict_channel,
                                        model_version_event_type=ModelVersionEventType.MODEL_DEPLOYED,
                                        dependency=predict_trigger, model_name=train_model.name)

    # workflow_id = af.run(example_util.get_project_path())
    # af.wait_workflow_execution_finished(workflow_id)
    # Run workflow
    stsp_flink_dag = 'stream_train_stream_predict_flink2'
    af.deploy_to_airflow(project_root_path, dag_id=stsp_flink_dag)
    context = af.run(project_path=project_root_path,
                     dag_id=stsp_flink_dag,
                     scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    run_job()
