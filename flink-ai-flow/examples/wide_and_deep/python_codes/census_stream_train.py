from flink_ml_tensorflow.tensorflow_TFConfig import TFConfig
from flink_ml_tensorflow.tensorflow_on_flink_mlconf import MLCONSTANTS
from flink_ml_tensorflow.tensorflow_on_flink_table import train
from pyflink.datastream.stream_execution_environment import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment


class CensusStreamTrain(object):

    @staticmethod
    def stream_train():
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        table_env = StreamTableEnvironment.create(stream_env)
        # table_env.get_config().get_configuration().set_string("pipeline.jars",
        #                                                       "file:///~/hadoop-2.7.7/share/hadoop/common/hadoop-common-2.7.7.jar;file:///~/hadoop-2.7.7/share/hadoop/hdfs/hadoop-hdfs-2.7.7.jar")
        #
        statement_set = table_env.create_statement_set()
        work_num = 2
        ps_num = 1
        python_file = "census_distribute.py"
        func = "stream_map_func"
        prop = {MLCONSTANTS.PYTHON_VERSION: '',
                MLCONSTANTS.ENCODING_CLASS: 'com.alibaba.flink.ml.operator.coding.RowCSVCoding',
                MLCONSTANTS.DECODING_CLASS: 'com.alibaba.flink.ml.operator.coding.RowCSVCoding',
                'sys:csv_encode_types': 'STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING,STRING',
                MLCONSTANTS.CONFIG_STORAGE_TYPE: MLCONSTANTS.STORAGE_ZOOKEEPER,
                MLCONSTANTS.CONFIG_ZOOKEEPER_CONNECT_STR: 'localhost:2181',
                MLCONSTANTS.CONFIG_ZOOKEEPER_BASE_PATH: '/demo',
                MLCONSTANTS.REMOTE_CODE_ZIP_FILE: "hdfs://localhost:9000/demo/code.zip"}
        env_path = None
        table_env.execute_sql("""
                    create table census_input_table (
                        adult varchar,
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
        input_tb = table_env.from_path('census_input_table')
        output_schema = None

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        train(stream_env, table_env, statement_set, input_tb, tf_config, output_schema)

        job_client = statement_set.execute().get_job_client()
        if job_client is not None:
            print(job_client.get_job_id())
            job_client.get_job_execution_result(user_class_loader=None).result()


if __name__ == "__main__":
    CensusStreamTrain.stream_train()
