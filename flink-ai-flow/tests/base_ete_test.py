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

import unittest
import os
import threading
from typing import Callable
import time
from airflow.events.scheduler_events import StopSchedulerEvent
from ai_flow.api.configuration import set_project_config_file
from airflow.contrib.jobs.event_based_scheduler_job import EventBasedSchedulerJob
from airflow.executors.local_executor import LocalExecutor
from ai_flow.application_master.master import AIFlowMaster
from notification_service.client import NotificationClient

from tests import db_utils


def project_path():
    return os.path.dirname(__file__)


def project_config_file():
    return project_path() + '/project.yaml'


def master_config_file():
    return project_path() + '/master.yaml'


def workflow_config_file():
    return project_path() + '/workflow.yaml'


master = AIFlowMaster(config_file=master_config_file())


def master_port():
    return master.master_config.get('master_port')


class BaseETETest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        master.start()

    @classmethod
    def tearDownClass(cls) -> None:
        master.stop()

    def setUp(self):
        master._clear_db()
        db_utils.clear_db_jobs()
        db_utils.clear_db_dags()
        db_utils.clear_db_serialized_dags()
        db_utils.clear_db_runs()
        db_utils.clear_db_task_execution()
        db_utils.clear_db_message()
        set_project_config_file(project_config_file())

    def tearDown(self):
        master._clear_db()

    @classmethod
    def start_scheduler(cls, file_path, executor=None):
        if executor is None:
            executor = LocalExecutor(3)

        scheduler = EventBasedSchedulerJob(
            dag_directory=file_path,
            server_uri="localhost:{}".format(master_port()),
            executor=executor,
            max_runs=-1,
            refresh_dag_dir_interval=30
        )
        print("scheduler starting")
        scheduler.run()

    def run_ai_flow(self, ai_flow_function: Callable[[], str], test_function: Callable[[NotificationClient], None],
                    executor=None):
        dag_file = ai_flow_function()

        def run_test_fun():
            time.sleep(5)
            client = NotificationClient(server_uri="localhost:{}".format(master_port()),
                                        default_namespace="test")
            test_function(client)
            client.send_event(StopSchedulerEvent(job_id=0).to_event())
        t = threading.Thread(target=run_test_fun, args=())
        t.setDaemon(True)
        t.start()
        self.start_scheduler(dag_file, executor)
