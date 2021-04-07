import os

import ai_flow as af
from ai_flow import ExampleSupportType, ModelType

import census_dataset


def get_project_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def init():
    af.register_example(
        name='batch_train_input',
        support_type=ExampleSupportType.EXAMPLE_BATCH,
        data_type='file',
        data_format='csv',
        batch_uri='/tmp/census_data/' + census_dataset.TRAINING_FILE)
    af.register_example(
        name='stream_train_input',
        support_type=ExampleSupportType.EXAMPLE_STREAM,
        data_type='file',
        data_format='csv',
        batch_uri='/tmp/census_data/' + census_dataset.TRAINING_FILE)
    af.register_example(
        name='stream_predict_input',
        support_type=ExampleSupportType.EXAMPLE_STREAM,
        data_type='kafka',
        data_format='csv',
        stream_uri='localhost:9092')
    af.register_example(
        name='stream_predict_output',
        support_type=ExampleSupportType.EXAMPLE_STREAM,
        data_type='kafka',
        data_format='csv',
        stream_uri='localhost:9092')
    af.register_model(
        model_name='wide_and_deep',
        model_type=ModelType.CHECKPOINT)


if __name__ == '__main__':
    af.set_project_config_file(get_project_path() + '/project.yaml')
    init()
