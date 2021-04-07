from typing import List

import numpy as np
import tensorflow as tf
from ai_flow import FunctionContext
from flink_ai_flow.pyflink import TableEnvCreator, SourceExecutor, FlinkFunctionContext, SinkExecutor, Executor
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings, Table, TableEnvironment, ScalarFunction, \
    DataTypes
from pyflink.table.udf import udf


class StreamTableEnvCreator(TableEnvCreator):

    def create_table_env(self):
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        stream_env.set_parallelism(1)
        t_env = StreamTableEnvironment.create(
            stream_env,
            environment_settings=EnvironmentSettings.new_instance().in_streaming_mode().use_blink_planner().build())
        statement_set = t_env.create_statement_set()
        t_env.get_config().set_python_executable('python3.7')
        t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)
        return stream_env, t_env, statement_set


class ReadOnlineExample(SourceExecutor):

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        table_env: TableEnvironment = function_context.get_table_env()
        table_env.execute_sql("""
            create table read_example (
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
                'properties.group.id' = 'read_example',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        """)
        table = table_env.from_path('read_example')
        return table


class Predict(ScalarFunction):

    def __init__(self):
        super().__init__()
        self._predictor = None
        self._exported_model = 'wide_and_deep/census_export/1604475484'

    def open(self, function_context: FunctionContext):
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


class PredictWideDeep(Executor):

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        function_context.t_env.register_function("predict",
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


class WriteOnlineExample(SinkExecutor):

    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        table_env: TableEnvironment = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        table_env.execute_sql("""
            create table write_example (
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
                'properties.group.id' = 'read_example',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        """)
        statement_set.add_insert('write_example', input_table)
