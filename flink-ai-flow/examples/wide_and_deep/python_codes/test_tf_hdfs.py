import os
import unittest

import ai_flow as af
import tensorflow as tf


class TFHdfs(unittest.TestCase):

    def setUp(self):
        # os.environ['$HADOOP_HDFS_HOME/lib/native'] = os.environ.get('HADOOP_HOME')
        # os.environ['LD_LIBRARY_PATH'] = '~/native-hadoop-library/hadoop-2.7.3/lib/native/'
        print(os.environ)

    def test_run_read_data(self):
        # file_name = "hdfs://localhost:9000/demo/input.csv"
        file_name = "/Users/chenwuchao/soft/apache/input.csv"
        # dataset = tf.data.TextLineDataset([file_name])
        dataset = tf.data.experimental.make_csv_dataset([file_name], batch_size=1)
        # dataset = dataset.repeat(3)
        it = dataset.make_one_shot_iterator()
        next = it.get_next()
        with tf.Session() as session:
            print(session.run(next))


code_path = os.path.dirname(__file__)

af.set_project_config_file(code_path + '/project.yaml')
af.register_model_version(model='wide_and_deep', model_path='|-----',
                          current_stage=af.ModelVersionStage.GENERATED)
