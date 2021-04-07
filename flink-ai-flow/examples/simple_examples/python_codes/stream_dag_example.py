#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
import os
import time
from typing import List

import ai_flow as af
from ai_flow import FunctionContext
from ai_flow.common.scheduler_type import SchedulerType
from notification_service.base_notification import EventWatcher, BaseEvent
from python_ai_flow import Executor


class PrintWatcher(EventWatcher):
    def __init__(self):
        self.events = []

    def process(self, events: List[BaseEvent]):
        self.events.extend(events)


class StreamPrintEventExecutor(Executor):
    def __init__(self, job_name):
        super().__init__()
        self.job_name = job_name
        self.watcher = PrintWatcher()

    def setup(self, function_context: FunctionContext):
        af.start_listen_event(key='key_1', watcher=self.watcher)

    def close(self, function_context: FunctionContext):
        af.stop_listen_event('key_1')

    def execute(self, function_context: FunctionContext, input_list: List) -> None:
        while True:
            if len(self.watcher.events) == 0:
                time.sleep(2)
            else:
                for event in self.watcher.events:
                    print(self.job_name + " " + str(event))
                    self.watcher.events.clear()


class SendEventExecutor(Executor):
    def __init__(self, key, value, event_type="UNDEFINED", num=1, pre_time=0, post_time=0):
        super().__init__()
        self.key = key
        self.value = value
        self.event_type = event_type
        self.num = num
        self.pre_time = pre_time
        self.post_time = post_time

    def execute(self, function_context: FunctionContext, input_list: List) -> None:
        for i in range(self.num):
            time.sleep(self.pre_time)
            af.send_event(self.key, self.value, self.event_type)
            time.sleep(self.post_time)


project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_workflow():
    with af.global_config_file(project_path + '/resources/workflow_config.yaml'):
        with af.config('job_1'):
            af.user_define_operation(af.PythonObjectExecutor(StreamPrintEventExecutor('job_1')))

        with af.config('job_2'):
            af.user_define_operation(af.PythonObjectExecutor(
                SendEventExecutor(key='key_1', value='value_1', num=5, post_time=5)))


def run_workflow():
    build_workflow()
    print(project_path)
    af.set_project_config_file(project_path + '/project.yaml')

    # res = af.run(project_path, dag_id='stream_dag_example', scheduler_type=SchedulerType.AIRFLOW)
    # af.wait_workflow_execution_finished(res)

    transform_dag = 'stream_dag_example22'
    af.deploy_to_airflow(project_path, dag_id=transform_dag)
    context = af.run(project_path=project_path,
                     dag_id=transform_dag,
                     scheduler_type=SchedulerType.AIRFLOW)


if __name__ == '__main__':
    run_workflow()
