from pyflink.dataset import ExecutionEnvironment
from pyflink.table import TableConfig, DataTypes, BatchTableEnvironment
from pyflink.table.descriptors import Schema, OldCsv, FileSystem
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import Table, DataTypes, ScalarFunction, CsvTableSink, WriteMode, TableEnvironment, \
    StreamTableEnvironment, EnvironmentSettings
from pyflink.table.udf import udf
from joblib import dump, load
import pandas as pd


def pyflink_job():
    EXAMPLE_COLUMNS = ['sl', 'sw', 'pl', 'pw', 'type']
    stream_env = StreamExecutionEnvironment.get_execution_environment()
    stream_env.set_parallelism(1)
    t_env = StreamTableEnvironment.create(
        stream_env,
        environment_settings=EnvironmentSettings.new_instance().use_blink_planner().build())
    # t_env.get_config().set_python_executable('/Users/kenken/py37/bin/python')
    t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)

    path = '/Users/kenken/Codes/flink-ai-extended/flink-ai-flow/examples/flink_examples/python_codes/flink_sklearn_example/dataset/iris_test.csv'

    out = '/Users/kenken/Codes/flink-ai-extended/flink-ai-flow/examples/flink_examples/python_codes/flink_sklearn_example/stream_train_stream_predict_pyflink/output2.csv'
    t_env.connect(FileSystem().path(path)) \
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

    model_path = '/Users/kenken/Codes/flink-ai-extended/flink-ai-flow/examples/flink_examples/python_codes/flink_sklearn_example/stream_train_stream_predict_pyflink/saved_model/2021-04-16-10:57:49'
    print(model_path)
    clf = load(model_path)

    class predict(ScalarFunction):
        def eval(self, sl, sw, pl, pw):
            records = [[sl, sw, pl, pw]]
            df = pd.DataFrame.from_records(records, columns=['sl', 'sw', 'pl', 'pw'])
            return clf.predict(df)[0]
            # return pw

    udf_func = udf(predict(), input_types=[DataTypes.FLOAT()] * 4, result_type=DataTypes.FLOAT())
    t_env.register_function('udf', udf_func)

    t_env.register_table_sink("write_example12345678", CsvTableSink(
        ['a'],
        [DataTypes.FLOAT()],
        out,
        write_mode=WriteMode.OVERWRITE
    ))
    t_env.from_path('mySource') \
        .select("udf(sl, sw, pl, pw) as r").insert_into('write_example12345678')
    t_env.execute("jobname")
if __name__ == '__main__':
    pyflink_job()