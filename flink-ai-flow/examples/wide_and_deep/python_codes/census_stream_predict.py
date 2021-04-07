import os

import ai_flow as af
from ai_flow import ExampleSupportType, ExampleMeta, ModelMeta, ModelType
from ai_flow.common.scheduler_type import SchedulerType
from ai_flow.graph.edge import TaskAction, MetCondition
from flink_ai_flow import LocalFlinkJobConfig, FlinkPythonExecutor
from python_ai_flow.local_python_job import LocalPythonJobConfig

from census_stream_executor import StreamTableEnvCreator, ReadOnlineExample, PredictWideDeep, \
    WriteOnlineExample
from common_executors import SendEventExecutor


def get_project_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_workflow():
    read_example_meta: ExampleMeta = af.register_example(
        name='census_read_example',
        support_type=ExampleSupportType.EXAMPLE_STREAM,
        data_type='kafka',
        data_format='csv',
        stream_uri='localhost:9092')
    write_example_meta: ExampleMeta = af.register_example(
        name='census_write_example',
        support_type=ExampleSupportType.EXAMPLE_STREAM,
        data_type='kafka',
        data_format='csv',
        stream_uri='localhost:9092')
    train_model_meta: ModelMeta = af.register_model(
        model_name='wide&deep',
        model_type=ModelType.SAVED_MODEL)
    return read_example_meta, write_example_meta, train_model_meta


def run_workflow():
    read_example_meta, write_example_meta, train_model_meta = prepare_workflow()

    job_config = LocalFlinkJobConfig()
    job_config.local_mode = 'local'
    job_config.flink_home = os.environ['FLINK_HOME']
    job_config.job_name = 'online_cluster'
    job_config.set_table_env_create_func(StreamTableEnvCreator())

    with af.config(job_config):
        read_example = af.read_example(example_info=read_example_meta,
                                       executor=FlinkPythonExecutor(python_object=ReadOnlineExample()))
        predict_model = af.predict(input_data_list=[read_example],
                                   model_info=train_model_meta,
                                   executor=FlinkPythonExecutor(python_object=PredictWideDeep()))
        af.write_example(input_data=predict_model,
                         example_info=write_example_meta,
                         executor=FlinkPythonExecutor(python_object=WriteOnlineExample()))
    python_config = LocalPythonJobConfig(job_name='job_1')
    with af.config(python_config):
        op_1 = af.user_define_operation(af.PythonObjectExecutor(
            SendEventExecutor(key='key_1', value='value_1', num=1)))

    python_config_2 = LocalPythonJobConfig(job_name='job_2')
    with af.config(python_config_2):
        op_2 = af.user_define_operation(af.PythonObjectExecutor(
            SendEventExecutor(key='key_2', value='value_2', num=1, pre_time=60)))

    af.user_define_control_dependency(predict_model, op_1, event_key='key_1', event_type='UNDEFINED',
                                      event_value="value_1", condition=MetCondition.SUFFICIENT, action=TaskAction.START)
    af.user_define_control_dependency(predict_model, op_2, event_key='key_2', event_type='UNDEFINED',
                                      event_value="value_2", condition=MetCondition.SUFFICIENT,
                                      action=TaskAction.RESTART)
    af.run(get_project_path(), dag_id='stream_flink', scheduler_type=SchedulerType.AIRFLOW)
    # workflow_id = af.run(get_project_path(), scheduler_type=SchedulerType.AIFLOW)
    # res = af.wait_workflow_execution_finished(workflow_id)
    # sys.exit(res)


if __name__ == '__main__':
    af.set_project_config_file(get_project_path() + '/project.yaml')
    run_workflow()
