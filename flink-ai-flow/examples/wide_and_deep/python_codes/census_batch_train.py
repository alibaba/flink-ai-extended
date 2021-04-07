from flink_ml_tensorflow.tensorflow_TFConfig import TFConfig
from flink_ml_tensorflow.tensorflow_on_flink_mlconf import MLCONSTANTS
from flink_ml_tensorflow.tensorflow_on_flink_table import train
from pyflink.datastream.stream_execution_environment import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment


class CensusBatchTrain(object):

    @staticmethod
    def batch_train():
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        table_env = StreamTableEnvironment.create(stream_env)
        statement_set = table_env.create_statement_set()
        work_num = 2
        ps_num = 1
        python_file = "census_distribute.py"
        func = "map_func"
        prop = {MLCONSTANTS.PYTHON_VERSION: '', MLCONSTANTS.CONFIG_STORAGE_TYPE: MLCONSTANTS.STORAGE_ZOOKEEPER,
                MLCONSTANTS.CONFIG_ZOOKEEPER_CONNECT_STR: 'localhost:2181',
                MLCONSTANTS.CONFIG_ZOOKEEPER_BASE_PATH: '/demo',
                MLCONSTANTS.REMOTE_CODE_ZIP_FILE: "hdfs://localhost:9000/demo/code.zip"}
        env_path = None

        input_tb = None
        output_schema = None

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        train(stream_env, table_env, statement_set, input_tb, tf_config, output_schema)

        job_client = statement_set.execute().get_job_client()
        if job_client is not None:
            job_client.get_job_execution_result(user_class_loader=None).result()


if __name__ == "__main__":
    CensusBatchTrain.batch_train()
