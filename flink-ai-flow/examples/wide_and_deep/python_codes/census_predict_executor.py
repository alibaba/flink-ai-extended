import os
from typing import List

import ai_flow as af
import numpy as np
import tensorflow as tf
from ai_flow import FunctionContext
from flink_ai_flow.pyflink import SourceExecutor, FlinkFunctionContext, SinkExecutor, Executor
from pyflink.table import Table, TableEnvironment, ScalarFunction, \
    DataTypes
from pyflink.table.udf import udf


class StreamPredictSource(SourceExecutor):

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        table_env: TableEnvironment = function_context.get_table_env()
        table_env.execute_sql('''
            create table stream_predict_source (
                age varchar,
                workclass varchar,
                fnlwgt varchar,
                education varchar,
                education_num varchar,
                marital_status varchar,
                occupation varchar,
                relationship varchar,
                race varchar,
                gender varchar,
                capital_gain varchar,
                capital_loss varchar,
                hours_per_week varchar,
                native_country varchar,
                income_bracket varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'census_input_topic',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'stream_predict_source',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        ''')
        table = table_env.from_path('stream_predict_source')
        return table


class Predict(ScalarFunction):

    def __init__(self):
        super().__init__()
        self._predictor = None
        self._exported_model = None

    def open(self, function_context: FunctionContext):
        af.set_project_config_file(os.path.dirname(__file__) + '/project.yaml')
        model_version = af.get_deployed_model_version("wide_and_deep")
        print("predict model path:" + model_version.model_path)
        self._exported_model = model_version.model_path.split('|')[1]
        with tf.Session() as session:
            tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], self._exported_model)
            self._predictor = tf.contrib.predictor.from_saved_model(self._exported_model)

    def eval(self, age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship,
             race, gender, capital_gain, capital_loss, hours_per_week, native_country):
        try:
            feature_dict = {
                'age': self._float_feature(value=int(age)),
                'workclass': self._bytes_feature(value=workclass.encode()),
                'fnlwgt': self._float_feature(value=int(fnlwgt)),
                'education': self._bytes_feature(value=education.encode()),
                'education_num': self._float_feature(value=int(education_num)),
                'marital_status': self._bytes_feature(value=marital_status.encode()),
                'occupation': self._bytes_feature(value=occupation.encode()),
                'relationship': self._bytes_feature(value=relationship.encode()),
                'race': self._bytes_feature(value=race.encode()),
                'gender': self._bytes_feature(value=gender.encode()),
                'capital_gain': self._float_feature(value=int(capital_gain)),
                'capital_loss': self._float_feature(value=int(capital_loss)),
                'hours_per_week': self._float_feature(value=float(hours_per_week)),
                'native_country': self._bytes_feature(value=native_country.encode()),
            }
            model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            model_input = model_input.SerializeToString()
            output_dict = self._predictor({'inputs': [model_input]})
            print(str(np.argmax(output_dict['scores'])))
            return str(np.argmax(output_dict['scores']))
        except Exception:
            return ''

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class StreamPredictExecutor(Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        function_context.t_env.register_function('predict',
                                                 udf(f=Predict(), input_types=[DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING(),
                                                                               DataTypes.STRING(), DataTypes.STRING()],
                                                     result_type=DataTypes.STRING()))
        return [input_list[0].select(
            'age, workclass, fnlwgt, education, education_num, marital_status, occupation, '
            'relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country, '
            'predict(age, workclass, fnlwgt, education, education_num, marital_status, occupation, '
            'relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country) as income_bracket')]


class StreamPredictSink(SinkExecutor):

    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        table_env: TableEnvironment = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        table_env.execute_sql('''
            create table stream_predict_sink (
                age varchar,
                workclass varchar,
                fnlwgt varchar,
                education varchar,
                education_num varchar,
                marital_status varchar,
                occupation varchar,
                relationship varchar,
                race varchar,
                gender varchar,
                capital_gain varchar,
                capital_loss varchar,
                hours_per_week varchar,
                native_country varchar,
                income_bracket varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'census_output_topic',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'stream_predict_sink',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        ''')
        statement_set.add_insert('stream_predict_sink', input_table)
