import os
import sys
import datetime

import tensorflow as tf
from flink_ml_tensorflow.tensorflow_context import TFContext


class FlinkReader(object):
    def __init__(self, context, batch_size=1, features={'input': tf.FixedLenFeature([], tf.string)}):
        self._context = context
        self._batch_size = batch_size
        self._features = features
        self._build_graph()

    def _decode(self, features):
        return features['input']

    def _build_graph(self):
        dataset = self._context.flink_stream_dataset()
        dataset = dataset.map(lambda record: tf.parse_single_example(record, features=self._features))
        dataset = dataset.map(self._decode)
        dataset = dataset.batch(self._batch_size)
        iterator = dataset.make_one_shot_iterator()
        self._next_batch = iterator.get_next()

    def next_batch(self, sess):
        try:
            batch = sess.run(self._next_batch)
            return batch
        except tf.errors.OutOfRangeError:
            return None


class FlinkWriter(object):
    def __init__(self, context):
        self._context = context
        self._build_graph()

    def _build_graph(self):
        self._write_feed = tf.placeholder(dtype=tf.string)
        self.write_op, self._close_op = self._context.output_writer_op([self._write_feed])

    def _example(self, results):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'output': tf.train.Feature(bytes_list=tf.train.BytesList(value=[results[0]])),
            }
        ))
        return example

    def write_result(self, sess, results):
        sess.run(self.write_op, feed_dict={self._write_feed: self._example(results).SerializeToString()})

    def close(self, sess):
        sess.run(self._close_op)


def test_source_sink(context):
    tf_context = TFContext(context)
    if 'ps' == tf_context.get_role_name():
        from time import sleep
        while True:
            sleep(1)
    else:
        index = tf_context.get_index()
        job_name = tf_context.get_role_name()
        cluster_json = tf_context.get_tf_cluster()
        cluster = tf.train.ClusterSpec(cluster=cluster_json)

        server = tf.train.Server(cluster, job_name=job_name, task_index=index)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                     device_filters=["/job:ps", "/job:worker/task:%d" % index])
        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:' + str(index), cluster=cluster)):
            # with flink_server_device(tf_context) as (device, server, sess_config):
            reader = FlinkReader(tf_context)
            writer = FlinkWriter(tf_context)

            with tf.train.ChiefSessionCreator(master=server.target, config=sess_config).create_session() as sess:
                while True:
                    batch = reader.next_batch(sess)
                    if batch is None:
                        break
                    tf.logging.info("[TF][%s]process %s" % (str(datetime.datetime.now()), str(batch)))

                    writer.write_result(sess, batch)
                writer.close(sess)
                sys.stdout.flush()
