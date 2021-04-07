import os
import sys
import time
import uuid

import yaml
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic


class CensusKafkaUtil(object):
    def __init__(self):
        super().__init__()
        self._yaml_config = None
        with open(os.path.dirname(os.path.abspath(__file__)) + '/kafka_config.yaml', 'r') as yaml_file:
            self._yaml_config = yaml.load(yaml_file)
        self.bootstrap_servers = self._yaml_config.get('bootstrap_servers')
        self.census_input_topic = self._yaml_config.get('census_input_topic')
        self.census_output_topic = self._yaml_config.get('census_output_topic')
        self.admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)

    def _send_data_loop(self, count=None):
        raw_data = []
        with open(file=self._yaml_config.get('dataset_uri'), mode='r') as f:
            for line in f.readlines():
                raw_data.append(line[:-1])
        producer = KafkaProducer(bootstrap_servers=[self.bootstrap_servers])
        num = 0
        if count is None:
            count = sys.maxsize
        while num < count:
            for line in raw_data:
                num += 1
                producer.send(self.census_input_topic,
                              key=bytes(str(uuid.uuid1()), encoding='utf8'),
                              value=bytes(line, encoding='utf8'))
                if num > count:
                    break
                if 0 == num % 1000:
                    print("send data {}".format(num))
                    time.sleep(self._yaml_config.get('time_interval') / 1000)

    def create_topic(self):
        topics = self.admin_client.list_topics()
        print(topics)
        if self.census_input_topic in topics:
            self.admin_client.delete_topics(topics=[self.census_input_topic], timeout_ms=5000)
            time.sleep(5)
        self.admin_client.create_topics(
            new_topics=[NewTopic(name=self.census_input_topic, num_partitions=1, replication_factor=1)])
        if self.census_output_topic in topics:
            self.admin_client.delete_topics(topics=[self.census_output_topic], timeout_ms=5000)
            time.sleep(5)
        self.admin_client.create_topics(
            new_topics=[NewTopic(name=self.census_output_topic, num_partitions=1, replication_factor=1)])
        # self._send_data_loop(count)

    def read_input_data(self, count):
        self.read_data(self.census_input_topic, count)

    def read_output_data(self, count):
        self.read_data(self.census_output_topic, count)

    def read_data(self, topic, count=None):
        consumer = KafkaConsumer(topic, bootstrap_servers=[self.bootstrap_servers], group_id=str(
            uuid.uuid1()), auto_offset_reset='earliest')
        num = 0
        if count is None:
            count = sys.maxsize
        for message in consumer:
            num += 1
            print(message.value)
            if num > count:
                break


if __name__ == '__main__':
    kafka_util = CensusKafkaUtil()
    kafka_util._send_data_loop(2000000000)
    # kafka_util.read_data(100)
    # kafka_util.create_topic()
    # kafka_util.read_input_data(100)
    # kafka_util.read_output_data(100)
