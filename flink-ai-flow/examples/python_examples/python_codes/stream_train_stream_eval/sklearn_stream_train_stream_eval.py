from ai_flow import ExampleSupportType, PythonObjectExecutor, ModelType

from ai_flow.common.path_util import get_file_dir
from ai_flow.common.scheduler_type import SchedulerType
from stream_train_stream_eval_executor import *
import example_util
import example_data_util

EXAMPLE_URI = example_data_util.get_example_data() + '/mnist_{}.npz'


def run_job():
    project_root_path = example_util.get_root_path()
    af.set_project_config_file(project_root_path + '/project.yaml')
    evaluate_trigger = af.external_trigger(name='evaluate')
    with af.engine('python'):
        project_meta = af.register_project(name='sklearn_stream_train_stream_predict_project',
                                           uri='../stream_train_stream_predict_example',
                                           project_type='local python')
        train_example = af.register_example(name='train_example',
                                            support_type=ExampleSupportType.EXAMPLE_STREAM,
                                            stream_uri=EXAMPLE_URI.format('train'),
                                            data_format='npz')
        train_read_example = af.read_example(example_info=train_example,
                                             executor=PythonObjectExecutor(python_object=ExampleTrain()))
        train_transform = af.transform(input_data_list=[train_read_example],
                                       executor=PythonObjectExecutor(python_object=TransformTrain()))
        train_model = af.register_model(model_name='logistic-regression',
                                        model_type=ModelType.SAVED_MODEL,
                                        model_desc='logistic regression model')
        train_channel = af.train(input_data_list=[train_transform],
                                 executor=PythonObjectExecutor(python_object=TrainModel()),
                                 model_info=train_model)



        evaluate_example = af.register_example(name='evaluate_example',
                                               support_type=ExampleSupportType.EXAMPLE_STREAM,
                                               stream_uri=EXAMPLE_URI.format('evaluate'),
                                               data_format='npz')
        evaluate_read_example = af.read_example(example_info=evaluate_example,
                                                executor=PythonObjectExecutor(python_object=ExampleEvaluate()))
        evaluate_transform = af.transform(input_data_list=[evaluate_read_example],
                                          executor=PythonObjectExecutor(python_object=TransformEvaluate()))
        print(get_file_dir(__file__) + '/evaluate_model')
        evaluate_artifact = af.register_artifact(name='evaluate_artifact',
                                                 stream_uri=get_file_dir(__file__) + '/evaluate_model')
        evaluate_channel = af.evaluate(input_data_list=[evaluate_transform],
                                       model_info=train_model,
                                       executor=PythonObjectExecutor(python_object=EvaluateModel()))

    print(train_model.name)
    af.model_version_control_dependency(src=evaluate_channel,
                                        model_version_event_type=ModelVersionEventType.MODEL_GENERATED,
                                        dependency=evaluate_trigger, model_name=train_model.name, namespace='scheduler')

    # Run workflow
    transform_dag = 'stream_train_stream_eval_air22'
    af.deploy_to_airflow(project_root_path, dag_id=transform_dag)
    context = af.run(project_path=project_root_path,
                     dag_id=transform_dag,
                     scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    run_job()
