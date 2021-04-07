from ai_flow.common.scheduler_type import SchedulerType
from flink_ai_flow import LocalFlinkJobConfig, FlinkPythonExecutor

from census_executor import *
from census_predict_executor import *


def get_project_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_workflow():
    with af.global_config_file(config_path=get_project_path() + '/resources/workflow_config.yaml'):
        stream_train_input = af.get_example_by_name('stream_train_input')
        stream_predict_input = af.get_example_by_name('stream_predict_input')
        stream_predict_output = af.get_example_by_name('stream_predict_output')
        model_info = af.get_model_by_name('wide_deep')

        workflow_config = af.default_af_job_context().global_workflow_config

        batch_train_config: LocalFlinkJobConfig = workflow_config.job_configs['census_batch_train']
        batch_train_config.set_table_env_create_func(StreamTableEnvCreator())

        stream_train_config: LocalFlinkJobConfig = workflow_config.job_configs['census_stream_train']
        stream_train_config.set_table_env_create_func(StreamTableEnvCreator())

        stream_predict_config: LocalFlinkJobConfig = workflow_config.job_configs['census_stream_predict']
        stream_predict_config.set_table_env_create_func(StreamTableEnvCreator())

        with af.config(config=batch_train_config):
            batch_train_channel = af.train(input_data_list=[],
                                           executor=FlinkPythonExecutor(python_object=BatchTrainExecutor()),
                                           model_info=model_info, name='census_batch_train')

        with af.config(config=stream_train_config):
            stream_train_source = af.read_example(example_info=stream_train_input,
                                                  executor=FlinkPythonExecutor(python_object=StreamTrainSource()))
            stream_train_channel = af.train(input_data_list=[stream_train_source],
                                              model_info=model_info,
                                              executor=FlinkPythonExecutor(python_object=StreamTrainExecutor()))

        with af.config(config=stream_predict_config):
            stream_predict_source = af.read_example(example_info=stream_predict_input,
                                                    executor=FlinkPythonExecutor(python_object=StreamPredictSource()))
            stream_predict_channel = af.predict(input_data_list=[stream_predict_source],
                                                model_info=model_info,
                                                executor=FlinkPythonExecutor(python_object=StreamPredictExecutor()))
            stream_predict_sink = af.write_example(input_data=stream_predict_channel,
                                                   example_info=stream_predict_output,
                                                   executor=FlinkPythonExecutor(python_object=StreamPredictSink()))

        af.stop_before_control_dependency(stream_train_channel, batch_train_channel)
        # af.stop_before_control_dependency(stream_predict_sink, batch_train_channel)
        af.model_version_control_dependency(src=stream_predict_sink, dependency=batch_train_channel,
                                            model_name='wide_and_deep', model_version_event_type='MODEL_DEPLOYED')

        # workflow_id = af.run(get_project_path(), scheduler_type=SchedulerType.AIFLOW)
        # af.wait_workflow_execution_finished(workflow_id)

        af.run(get_project_path(), 'census_workflow', scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    af.set_project_config_file(get_project_path() + '/project.yaml')
    run_workflow()
