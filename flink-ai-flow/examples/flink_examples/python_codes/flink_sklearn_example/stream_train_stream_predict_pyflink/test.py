# from pyflink.dataset import ExecutionEnvironment
# from pyflink.table import TableConfig, DataTypes, BatchTableEnvironment
# from pyflink.table.descriptors import Schema, OldCsv, FileSystem
#
# exec_env = ExecutionEnvironment.get_execution_environment()
# exec_env.set_parallelism(1)
# t_config = TableConfig()
# t_env = BatchTableEnvironment.create(exec_env, t_config)
#
# pp = '/Users/kenken/Codes/flink-ai-extended/flink-ai-flow/flink_ai_flow/tests/resources/word_count.txt'
# t_env.connect(FileSystem().path(pp)) \
#     .with_format(OldCsv()
#                  .field('word', DataTypes.STRING())) \
#     .with_schema(Schema()
#                  .field('word', DataTypes.STRING())) \
#     .create_temporary_table('mySource')
# oo = '/Users/kenken/Codes/flink-ai-extended/flink-ai-flow/examples/flink_examples/python_codes/flink_sklearn_example/stream_train_stream_predict_pyflink/output.csv'
# t_env.connect(FileSystem().path(oo)) \
#     .with_format(OldCsv()
#                  .field_delimiter('\t')
#                  .field('word', DataTypes.STRING())
#                  .field('count', DataTypes.BIGINT())) \
#     .with_schema(Schema()
#                  .field('word', DataTypes.STRING())
#                  .field('count', DataTypes.BIGINT())) \
#     .create_temporary_table('mySink')
#
# t_env.from_path('mySource') \
#     .group_by('word') \
#     .select('word, count(1)') \
#     .insert_into('mySink')
#
# t_env.execute("tutorial_job")