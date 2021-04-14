from ai_flow import ExampleSupportType, ExecuteArgs, PythonObjectExecutor, ModelType
from ai_flow.application_master.master import AIFlowMaster
from ai_flow.common.args import Args
from ai_flow.common.path_util import get_file_dir
from ai_flow.common.scheduler_type import SchedulerType
import example_util
from flink_ai_flow import FlinkJavaExecutor
from flink_ai_flow import LocalFlinkJobConfig
from flink_ai_flow.pyflink.user_define_executor import TableEnvCreator
from batch_train_batch_predict_executor import *

EXAMPLE_DATASET = get_file_dir(__file__) + '/dataset/{}.csv'


def run_job():
    project_root_path = example_util.get_root_path()
    af.send_event()
    af.set_project_config_file(project_root_path + '/project.yaml')
    job_config = LocalFlinkJobConfig()
    """configure execution mode of job config"""
    job_config.exec_mode = af.ExecutionMode.BATCH
    with af.config(job_config):
        project_meta = af.register_project(name='sklearn_batch_train_batch_predict_project',
                                           uri='../flink_sklearn_example',
                                           project_type='local python')
        import_example = af.register_example(name='import_example',
                                             support_type=ExampleSupportType.EXAMPLE_BATCH,
                                             batch_uri=EXAMPLE_DATASET.format('winequality-red'),
                                             data_format='csv',
                                             data_type='pandas')
        transform_example = af.read_example(example_info=import_example,
                                            exec_args=ExecuteArgs(
                                                batch_properties=read_example_properties(import_example.batch_uri)),
                                            executor=PythonObjectExecutor(python_object=TableEnvCreator()))
        train_transform = af.transform(input_data_list=[transform_example],
                                       executor=FlinkJavaExecutor(
                                           java_class='com.apache.flink.ai.flow.examples.SklearnTransformer'))
        train_example = af.register_example(name='train_example',
                                            support_type=ExampleSupportType.EXAMPLE_BATCH,
                                            batch_uri=EXAMPLE_DATASET.format('winequality-red-train'),
                                            data_type='pandas',
                                            data_format='csv')
        export_example = af.write_example(input_data=train_transform,
                                          example_info=train_example,
                                          exec_args=ExecuteArgs(
                                              batch_properties=write_example_properties(train_example.batch_uri)))

    with af.engine('python'):
        train_read_example = af.read_example(example_info=train_example,
                                             exec_args=ExecuteArgs(
                                                 batch_properties=Args(header=None, names=EXAMPLE_COLUMNS)))
        train_model = af.register_model(model_name='elastic-net',
                                        model_type=ModelType.SAVED_MODEL,
                                        model_desc='elastic net model')
        train_channel = af.train(input_data_list=[train_read_example],
                                 executor=PythonObjectExecutor(python_object=TrainModel()),
                                 model_info=train_model)

        evaluate_example = af.register_example(name='evaluate_example',
                                               support_type=ExampleSupportType.EXAMPLE_BATCH,
                                               batch_uri=EXAMPLE_DATASET.format('winequality-red-train'),
                                               data_type='pandas',
                                               data_format='csv')
        evaluate_read_example = af.read_example(example_info=evaluate_example,
                                                exec_args=ExecuteArgs(
                                                    batch_properties=Args(header=None, names=EXAMPLE_COLUMNS)))
        evaluate_artifact = af.register_artifact(name='evaluate_artifact',
                                                 batch_uri=get_file_dir(
                                                     __file__) + '/evaluate_model')
        evaluate_channel = af.evaluate(input_data_list=[evaluate_read_example],
                                       model_info=train_model,
                                       executor=PythonObjectExecutor(python_object=EvaluateModel()))

        validate_example = af.register_example(name='validate_example',
                                               support_type=ExampleSupportType.EXAMPLE_BATCH,
                                               batch_uri=EXAMPLE_DATASET.format('winequality-red-train'),
                                               data_type='pandas',
                                               data_format='csv')
        validate_read_example = af.read_example(example_info=validate_example,
                                                exec_args=ExecuteArgs(
                                                    batch_properties=Args(header=None, names=EXAMPLE_COLUMNS)))
        validate_artifact = af.register_artifact(name='validate_artifact',
                                                 batch_uri=get_file_dir(
                                                     __file__) + '/validate_model')
        validate_channel = af.model_validate(input_data_list=[validate_read_example],
                                             model_info=train_model,
                                             executor=PythonObjectExecutor(python_object=ValidateModel()))

        predict_example = af.register_example(name='predict_example',
                                              support_type=ExampleSupportType.EXAMPLE_BATCH,
                                              batch_uri=EXAMPLE_DATASET.format('winequality-red-train'),
                                              data_type='pandas',
                                              data_format='csv')
        predict_read_example = af.read_example(example_info=predict_example,
                                               exec_args=ExecuteArgs(
                                                   batch_properties=Args(header=None, names=EXAMPLE_COLUMNS)))
        predict_channel = af.predict(input_data_list=[predict_read_example],
                                     model_info=train_model,
                                     executor=PythonObjectExecutor(python_object=PredictModel()))

        write_example = af.register_example(name='write_example',
                                            support_type=ExampleSupportType.EXAMPLE_BATCH,
                                            batch_uri=get_file_dir(
                                                __file__) + '/predict_model.txt',
                                            data_format='txt')
        af.write_example(input_data=predict_channel,
                         example_info=write_example,
                         executor=PythonObjectExecutor(python_object=WriteExample()))

    af.stop_before_control_dependency(train_read_example, export_example)
    af.stop_before_control_dependency(evaluate_channel, train_channel)
    af.stop_before_control_dependency(validate_channel, evaluate_channel)
    af.stop_before_control_dependency(predict_channel, validate_channel)


    # Run workflow
    transform_dag = 'batch_train_batch_predict_flink'
    af.deploy_to_airflow(project_root_path, dag_id=transform_dag)
    context = af.run(project_path=project_root_path,
                     dag_id=transform_dag,
                     scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    config_file = example_util.get_master_config_file()
    run_job()
