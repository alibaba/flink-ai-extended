import json
import os
from ai_flow.common.time_utils import generate_time_str
from joblib import dump, load
from time import sleep
import time

from ai_flow import Properties, ExampleMeta, ModelMeta, ModelVersionStage
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import Table, DataTypes, ScalarFunction, CsvTableSink, WriteMode, TableEnvironment, \
    StreamTableEnvironment, EnvironmentSettings
from pyflink.table.descriptors import FileSystem, OldCsv, Schema
from pyflink.table.udf import udf
from python_ai_flow import ExampleExecutor
from typing import List
import threading
import pandas as pd
import ai_flow as af
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from streamz import Stream

from flink_ai_flow.pyflink.user_define_executor import Executor, SourceExecutor, FlinkFunctionContext, SinkExecutor, \
    TableEnvCreator
from python_ai_flow import FunctionContext, Executor as PythonExecutor

EXAMPLE_COLUMNS = ['sl', 'sw', 'pl', 'pw', 'type']


def read_example_properties(stream_uri):
    properties: Properties = {}
    properties['connector.type'] = 'filesystem'
    properties['connector.path'] = stream_uri
    properties['connector.property-version'] = '1'
    properties['format.type'] = 'csv'
    properties['format.property-version'] = '1'
    properties['format.ignore-first-line'] = True

    properties['format.fields.0.type'] = 'FLOAT'
    properties['format.fields.1.type'] = 'FLOAT'
    properties['format.fields.2.type'] = 'FLOAT'
    properties['format.fields.3.type'] = 'FLOAT'
    properties['format.fields.4.type'] = 'INT'

    properties['format.fields.0.name'] = 'sl'
    properties['format.fields.1.name'] = 'sw'
    properties['format.fields.2.name'] = 'pl'
    properties['format.fields.3.name'] = 'pw'
    properties['format.fields.4.name'] = 'type'

    properties['schema.0.type'] = 'FLOAT'
    properties['schema.1.type'] = 'FLOAT'
    properties['schema.2.type'] = 'FLOAT'
    properties['schema.3.type'] = 'FLOAT'
    properties['schema.4.type'] = 'INT'

    properties['schema.0.name'] = 'sl'
    properties['schema.1.name'] = 'sw'
    properties['schema.2.name'] = 'pl'
    properties['schema.3.name'] = 'pw'
    properties['schema.4.name'] = 'type'
    return properties


def write_example_properties(stream_uri):
    properties: Properties = {}
    properties['connector.type'] = 'filesystem'
    properties['connector.path'] = stream_uri
    properties['connector.property-version'] = '1'
    properties['format.type'] = 'csv'
    properties['format.property-version'] = '1'
    properties['format.fields.0.type'] = 'FLOAT'
    properties['format.fields.1.type'] = 'FLOAT'
    properties['format.fields.2.type'] = 'FLOAT'
    properties['format.fields.3.type'] = 'FLOAT'
    properties['format.fields.4.type'] = 'INT'

    properties['format.fields.0.name'] = 'sl'
    properties['format.fields.1.name'] = 'sw'
    properties['format.fields.2.name'] = 'pl'
    properties['format.fields.3.name'] = 'pw'
    properties['format.fields.4.name'] = 'type'

    properties['schema.0.type'] = 'FLOAT'
    properties['schema.1.type'] = 'FLOAT'
    properties['schema.2.type'] = 'FLOAT'
    properties['schema.3.type'] = 'FLOAT'
    properties['schema.4.type'] = 'INT'

    properties['schema.0.name'] = 'sl'
    properties['schema.1.name'] = 'sw'
    properties['schema.2.name'] = 'pl'
    properties['schema.3.name'] = 'pw'
    properties['schema.4.name'] = 'type'
    return properties
#
# class StreamTrainSource(SourceExecutor):
#
#     def execute(self, function_context: FlinkFunctionContext) -> Table:
#         table_env: TableEnvironment = function_context.get_table_env()
#         table_env.execute_sql('''
#             create table stream_train_source (
#                 age varchar,
#                 workclass varchar,
#                 fnlwgt varchar,
#                 education varchar,
#                 education_num varchar,
#                 marital_status varchar,
#                 occupation varchar,
#                 relationship varchar,
#                 race varchar,
#                 gender varchar,
#                 capital_gain varchar,
#                 capital_loss varchar,
#                 hours_per_week varchar,
#                 native_country varchar,
#                 income_bracket varchar
#             ) with (
#                 'connector' = 'kafka',
#                 'topic' = 'census_input_topic',
#                 'properties.bootstrap.servers' = 'localhost:9092',
#                 'properties.group.id' = 'stream_train_source',
#                 'format' = 'csv',
#                 'scan.startup.mode' = 'earliest-offset'
#             )
#         ''')
#         table = table_env.from_path('stream_train_source')
#         return table

class ReadTrainExample:
    class SourceThread(threading.Thread):
        def __init__(self, stream_uri):
            super().__init__()
            self.stream = Stream()
            self.stream_uri = stream_uri

        def run(self) -> None:
            for i in range(0, 5):
                train_data = pd.read_csv(self.stream_uri, header=None, names=EXAMPLE_COLUMNS)
                y_train = train_data.pop(EXAMPLE_COLUMNS[4])
                self.stream.emit((train_data.values, y_train.values))
                print('send-{}-message'.format(i))
                sleep(20)

    class LoadTrainExample(ExampleExecutor):

        def setup(self, function_context: FunctionContext):
            example_meta: ExampleMeta = function_context.node_spec.example_meta
            self.thread = ReadTrainExample.SourceThread(example_meta.stream_uri)
            self.thread.start()

        def execute(self, function_context: FunctionContext, input_list: List) -> List:
            return [self.thread.stream]


class TrainModel(PythonExecutor):

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        def sink(df):
            pass

        def train(df):
            print('model update!')
            x_train, y_train = df[0], df[1]
            sk_model = KNeighborsClassifier(n_neighbors=5)
            sk_model.fit(x_train, y_train)
            model_meta: ModelMeta = function_context.node_spec.output_model
            workflow_execution_id = function_context.job_context.workflow_execution_id
            model_path = 'saved_model'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_version = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
            model_path = model_path + '/' + model_version
            dump(sk_model, model_path)
            af.register_model_version(model=model_meta, model_path=model_path)

            return df

        data: Stream = input_list[0]
        data.map(train).sink(sink)
        return []

class EvaluateExampleReader(Executor):
    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        train_data = pd.read_csv(function_context.node_spec.example_meta.stream_uri, header=None, names=EXAMPLE_COLUMNS)
        y_train = train_data.pop(EXAMPLE_COLUMNS[4])
        return [[train_data, y_train]]

class EvaluateModel(PythonExecutor):

    def __init__(self):
        super().__init__()
        self.model = None

    def setup(self, function_context: FunctionContext):
        pass

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print('evaluate' + generate_time_str())
        self.model = af.get_latest_generated_model_version(function_context.node_spec.model.name)
        evaluate_example = input_list[0]
        y_evaluate = evaluate_example.pop(EXAMPLE_COLUMNS[4])
        clf = load(self.model.model_path)
        scores = cross_val_score(clf, evaluate_example, y_evaluate, cv=5)
        evaluate_artifact = af.get_artifact_by_name('evaluate_artifact9').stream_uri
        print(scores)
        with open(evaluate_artifact, 'a') as f:
            f.write('model version: {} scores: {}\n'.format(self.model.version, scores))
        return []


class ValidateModel(PythonExecutor):

    def __init__(self):
        super().__init__()
        self.model_path = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        self.model = af.get_latest_generated_model_version(function_context.node_spec.model.name)

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        new_model_version = self.model_version
        model_meta: ModelMeta = function_context.node_spec.model
        deployed_model_version = af.get_deployed_model_version(model_name=model_meta.name)
        if deployed_model_version is None:
            af.update_model_version(model_name=model_meta.name, model_version=new_model_version,
                                    current_stage=ModelVersionStage.DEPLOYED)
            # af.update_notification(key='best_model_version',
            #                        value=new_model_version)
        else:
            evaluate_example = input_list[0]
            y_validate = evaluate_example.pop(EXAMPLE_COLUMNS[4])
            x_validate = evaluate_example
            knn = load(self.model_path)
            scores = knn.score(x_validate, y_validate)
            deployed_knn = load(deployed_model_version.model_path)
            deployed_scores = deployed_knn.score(x_validate, y_validate)
            stream_uri = af.get_artifact_by_name('validate_artifact9').stream_uri
            with open(stream_uri, 'a') as f:
                f.write('deployed model version: {} scores: {}\n'.format(deployed_model_version.version, deployed_scores))
                f.write('generated model version: {} scores: {}\n'.format(self.model_version, scores))
            if scores >= deployed_scores:
                af.update_model_version(model_name=model_meta.name,
                                        model_version=deployed_model_version.version,
                                        current_stage=ModelVersionStage.VALIDATED)
                af.update_model_version(model_name=model_meta.name,
                                        model_version=new_model_version,
                                        current_stage=ModelVersionStage.DEPLOYED)
                # af.update_notification(key='best_model_version',
                #                        value=new_model_version)
        return []


class StreamTableEnvCreator(TableEnvCreator):

    def create_table_env(self) -> TableEnvironment:
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_parallelism(1)
        return StreamTableEnvironment.create(stream_execution_environment=env,
                                             environment_settings=EnvironmentSettings.new_instance().use_blink_planner().build())


class Source(SourceExecutor):
    def execute(self, function_context: FlinkFunctionContext) -> Table:
        example_meta: ExampleMeta = function_context.get_example_meta()
        t_env = function_context.get_table_env()
        t_env.connect(FileSystem().path(example_meta.batch_uri)) \
            .with_format(OldCsv()
                         .ignore_first_line()
                         .field(EXAMPLE_COLUMNS[0], DataTypes.FLOAT())
                         .field(EXAMPLE_COLUMNS[1], DataTypes.FLOAT())
                         .field(EXAMPLE_COLUMNS[2], DataTypes.FLOAT())
                         .field(EXAMPLE_COLUMNS[3], DataTypes.FLOAT())) \
            .with_schema(Schema()
                         .field(EXAMPLE_COLUMNS[0], DataTypes.FLOAT())
                         .field(EXAMPLE_COLUMNS[1], DataTypes.FLOAT())
                         .field(EXAMPLE_COLUMNS[2], DataTypes.FLOAT())
                         .field(EXAMPLE_COLUMNS[3], DataTypes.FLOAT())) \
            .create_temporary_table('mySource')
        return t_env.from_path('mySource')


class Transformer(Executor):
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        self.model_name = function_context.node_spec.model.name
        while af.get_deployed_model_version(self.model_name) is None:
            time.sleep(5)
        print("### {} setup done for {}".format(self.__class__.__name__, function_context.node_spec.model.name))

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        model_meta = af.get_deployed_model_version(self.model_name)
        model_path = model_meta.model_path
        print(model_path)
        clf = load(model_path)

        class predict(ScalarFunction):
            def eval(self, sl, sw, pl, pw):
                records = [[sl, sw, pl, pw]]
                df = pd.DataFrame.from_records(records, columns=['sl', 'sw', 'pl', 'pw'])
                return clf.predict(df)[0]

        udf_func = udf(predict(), input_types=[DataTypes.FLOAT()] * 4, result_type=DataTypes.FLOAT())
        function_context.get_table_env().register_function('udf', udf_func)
        function_context.get_table_env().get_config().set_python_executable('/Users/kenken/tfenve/bin/python')
        return [input_list[0].select("udf(sl,sw,pl,pw) as r")]


class Sink(SinkExecutor):

    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        table_env = function_context.get_table_env()
        table_env.register_table_sink("write_example", CsvTableSink(
            ['a'],
            [DataTypes.FLOAT()],
            function_context.get_example_meta().batch_uri,
            write_mode=WriteMode.OVERWRITE
        ))
        input_table.insert_into('write_example')
