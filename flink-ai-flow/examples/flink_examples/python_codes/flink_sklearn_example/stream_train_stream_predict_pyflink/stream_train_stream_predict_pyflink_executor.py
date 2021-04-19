import json
import os
from ai_flow.common.time_utils import generate_time_str
from joblib import dump, load
from time import sleep
import time
from ai_flow.common.path_util import get_file_dir
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


class ReadTrainExample:
    class SourceThread(threading.Thread):
        def __init__(self, stream_uri):
            super().__init__()
            self.stream = Stream()
            self.stream_uri = stream_uri

        def run(self) -> None:
            for i in range(0, 5):
                train_data = pd.read_csv(self.stream_uri, header=0, names=EXAMPLE_COLUMNS)
                y_train = train_data.pop(EXAMPLE_COLUMNS[4])
                self.stream.emit((train_data.values, y_train.values))
                print('send-{}-message'.format(i))
                sleep(50)

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

            x_train, y_train = df[0], df[1]
            sk_model = KNeighborsClassifier(n_neighbors=5)
            sk_model.fit(x_train, y_train)
            model_meta = function_context.node_spec.output_model
            model_path = get_file_dir(__file__) + '/saved_model'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_version = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
            model_path = model_path + '/' + model_version
            dump(sk_model, model_path)
            af.register_model_version(model=model_meta, model_path=model_path)
            print('model update!')
            return df

        data: Stream = input_list[0]
        data.map(train).sink(sink)
        return []


class EvaluateExampleReader(Executor):
    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        train_data = pd.read_csv(function_context.node_spec.example_meta.stream_uri, header=0, names=EXAMPLE_COLUMNS)
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
        model_name = function_context.node_spec.model.name
        self.model = af.get_latest_generated_model_version(model_name)
        evaluate_example = input_list[0][0]
        print(evaluate_example)
        # y_evaluate = evaluate_example.pop(EXAMPLE_COLUMNS[4])
        y_evaluate = input_list[0][1]
        print("############")
        print(y_evaluate)
        clf = load(self.model.model_path)
        scores = cross_val_score(clf, evaluate_example, y_evaluate, cv=5)
        evaluate_artifact = af.get_artifact_by_name('evaluate_artifact').stream_uri
        print(scores)
        with open(evaluate_artifact, 'a') as f:
            f.write('model version: {} scores: {}\n'.format(self.model.version, scores))
        af.update_model_version(model_name=model_name, model_version=self.model.version,
                                current_stage=ModelVersionStage.DEPLOYED)
        return []


# class ValidateModel(PythonExecutor):
#
#     def __init__(self):
#         super().__init__()
#         self.model_path = None
#         self.model_version = None
#
#     def setup(self, function_context: FunctionContext):
#         self.model = af.get_latest_generated_model_version(function_context.node_spec.model.name)
#
#     def execute(self, function_context: FunctionContext, input_list: List) -> List:
#         new_model_version = self.model_version
#         model_meta: ModelMeta = function_context.node_spec.model
#         deployed_model_version = af.get_deployed_model_version(model_name=model_meta.name)
#         if deployed_model_version is None:
#             af.update_model_version(model_name=model_meta.name, model_version=new_model_version,
#                                     current_stage=ModelVersionStage.DEPLOYED)
#             # af.update_notification(key='best_model_version',
#             #                        value=new_model_version)
#         else:
#             evaluate_example = input_list[0]
#             y_validate = evaluate_example.pop(EXAMPLE_COLUMNS[4])
#             x_validate = evaluate_example
#             knn = load(self.model_path)
#             scores = knn.score(x_validate, y_validate)
#             deployed_knn = load(deployed_model_version.model_path)
#             deployed_scores = deployed_knn.score(x_validate, y_validate)
#             stream_uri = af.get_artifact_by_name('validate_artifact').stream_uri
#             with open(stream_uri, 'a') as f:
#                 f.write('deployed model version: {} scores: {}\n'.format(deployed_model_version.version, deployed_scores))
#                 f.write('generated model version: {} scores: {}\n'.format(self.model_version, scores))
#             if scores >= deployed_scores:
#                 af.update_model_version(model_name=model_meta.name,
#                                         model_version=deployed_model_version.version,
#                                         current_stage=ModelVersionStage.VALIDATED)
#                 af.update_model_version(model_name=model_meta.name,
#                                         model_version=new_model_version,
#                                         current_stage=ModelVersionStage.DEPLOYED)
#                 # af.update_notification(key='best_model_version',
#                 #                        value=new_model_version)
#         return []
#

class StreamTableEnvCreator(TableEnvCreator):

    def create_table_env(self):
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        stream_env.set_parallelism(1)
        t_env = StreamTableEnvironment.create(
            stream_env,
            environment_settings=EnvironmentSettings.new_instance().use_blink_planner().build())
        statement_set = t_env.create_statement_set()
        t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)
        return stream_env, t_env, statement_set


class Source(SourceExecutor):
    def execute(self, function_context: FlinkFunctionContext) -> Table:
        print("### {} setup done2 for {}".format(self.__class__.__name__, "sads"))
        example_meta: ExampleMeta = function_context.get_example_meta()
        t_env = function_context.get_table_env()
        print(example_meta.batch_uri)
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

        print("### {} execute done for {}".format(self.__class__.__name__, "sads"))
        # print(t_env.from_path('mySource').to_pandas())
        return t_env.from_path('mySource')


class Transformer(Executor):
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.model_version = None

    def setup(self, function_context: FunctionContext):
        self.model_name = function_context.node_spec.model.name
        # print(self.model_name)
        while af.get_deployed_model_version(self.model_name) is None:
            time.sleep(5)
        print("### {} setup done for {}".format(self.__class__.__name__, "ddd"))

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
        return [input_list[0].select("udf(sl,sw,pl,pw)")]


class Sink(SinkExecutor):

    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        print("### {} setup done".format(self.__class__.__name__))
        table_env = function_context.get_table_env()
        table_env.register_table_sink("write_example", CsvTableSink(
            ['a'],
            [DataTypes.FLOAT()],
            function_context.get_example_meta().batch_uri,
            write_mode=WriteMode.OVERWRITE
        ))
        input_table.print_schema()
        function_context.statement_set.add_insert("write_example", input_table)
        print("### {} table_env execute done {}".format(self.__class__.__name__, function_context.get_example_meta().batch_uri))

    def close(self, function_context: FlinkFunctionContext):
        print("Close done")