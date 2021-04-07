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
import os
import pickle
import sqlalchemy
import time
import unittest
import threading
import logging

from airflow.utils.state import State
from typing import List

import psutil
from airflow.executors.scheduling_action import SchedulingAction
from airflow.contrib.jobs.scheduler_client import EventSchedulerClient
from airflow.models.taskexecution import TaskExecution
from notification_service.base_notification import BaseEvent, UNDEFINED_EVENT_TYPE
from notification_service.client import NotificationClient
from notification_service.event_storage import MemoryEventStorage
from notification_service.master import NotificationMaster
from notification_service.service import NotificationService

from airflow.contrib.jobs.event_based_scheduler_job import EventBasedSchedulerJob, SchedulerEventWatcher, \
    EventBasedScheduler
from airflow.executors.local_executor import LocalExecutor
from airflow.models import TaskInstance, Message
from airflow.jobs.base_job import BaseJob
from airflow.utils.mailbox import Mailbox
from airflow.utils.session import create_session, provide_session
from airflow.events.scheduler_events import StopSchedulerEvent, StopDagEvent
from tests.test_utils import db

TEST_DAG_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), os.pardir, 'dags')
EVENT_BASED_SCHEDULER_DAG = 'event_based_scheduler_dag'


class TestEventBasedScheduler(unittest.TestCase):

    def setUp(self):
        db.clear_db_jobs()
        db.clear_db_dags()
        db.clear_db_serialized_dags()
        db.clear_db_runs()
        db.clear_db_task_execution()
        db.clear_db_message()
        self.scheduler = None
        self.port = 50103
        self.storage = MemoryEventStorage()
        self.master = NotificationMaster(NotificationService(self.storage), self.port)
        self.master.run()
        self.client = NotificationClient(server_uri="localhost:{}".format(self.port),
                                         default_namespace="test_namespace")
        time.sleep(1)

    def tearDown(self):
        self.master.stop()

    def _get_task_instance(self, dag_id, task_id, session):
        return session.query(TaskInstance).filter(
            TaskInstance.dag_id == dag_id,
            TaskInstance.task_id == task_id
        ).first()

    def schedule_task_function(self):
        stopped = False
        while not stopped:
            with create_session() as session:
                ti_sleep_1000_secs = self._get_task_instance(EVENT_BASED_SCHEDULER_DAG, 'sleep_1000_secs', session)
                ti_python_sleep = self._get_task_instance(EVENT_BASED_SCHEDULER_DAG, 'python_sleep', session)
                if ti_sleep_1000_secs and ti_sleep_1000_secs.state == State.SCHEDULED and \
                   ti_python_sleep and ti_python_sleep.state == State.SCHEDULED:
                    self.client.send_event(BaseEvent(key='start', value='', event_type='', namespace='test_namespace'))

                    while not stopped:
                        ti_sleep_1000_secs.refresh_from_db()
                        ti_python_sleep.refresh_from_db()
                        if ti_sleep_1000_secs and ti_sleep_1000_secs.state == State.RUNNING and \
                           ti_python_sleep and ti_python_sleep.state == State.RUNNING:
                            time.sleep(10)
                            break
                        else:
                            time.sleep(1)
                    self.client.send_event(BaseEvent(key='stop', value='',
                                                     event_type=UNDEFINED_EVENT_TYPE, namespace='test_namespace'))
                    self.client.send_event(BaseEvent(key='restart', value='',
                                                     event_type=UNDEFINED_EVENT_TYPE, namespace='test_namespace'))
                    while not stopped:
                        ti_sleep_1000_secs.refresh_from_db()
                        ti_python_sleep.refresh_from_db()
                        if ti_sleep_1000_secs and ti_sleep_1000_secs.state == State.KILLED and \
                           ti_python_sleep and ti_python_sleep.state == State.RUNNING:
                            stopped = True
                        else:
                            time.sleep(1)
                else:
                    time.sleep(1)
        self.client.send_event(StopSchedulerEvent(job_id=0).to_event())

    def test_event_based_scheduler(self):
        t = threading.Thread(target=self.schedule_task_function)
        t.setDaemon(True)
        t.start()
        self.start_scheduler('../../dags/test_event_based_scheduler.py')

    def test_replay_message(self):
        key = "stop"
        mailbox = Mailbox()
        mailbox.set_scheduling_job_id(1234)
        watcher = SchedulerEventWatcher(mailbox)
        self.client.start_listen_events(
            watcher=watcher,
            start_time=int(time.time() * 1000),
            version=None
        )
        self.send_event(key)
        msg: BaseEvent = mailbox.get_message()
        self.assertEqual(msg.key, key)
        with create_session() as session:
            msg_from_db = session.query(Message).first()
            expect_non_unprocessed = EventBasedScheduler.get_unprocessed_message(1000)
            self.assertEqual(0, len(expect_non_unprocessed))
            unprocessed = EventBasedScheduler.get_unprocessed_message(1234)
            self.assertEqual(unprocessed[0].serialized_message, msg_from_db.data)
        deserialized_data = pickle.loads(msg_from_db.data)
        self.assertEqual(deserialized_data.key, key)
        self.assertEqual(msg, deserialized_data)

    def send_event(self, key):
        event = self.client.send_event(BaseEvent(key=key,
                                                 event_type=UNDEFINED_EVENT_TYPE,
                                                 value="value1"))
        self.assertEqual(key, event.key)

    @provide_session
    def get_task_execution(self, dag_id, task_id, session):
        return session.query(TaskExecution).filter(TaskExecution.dag_id == dag_id,
                                                   TaskExecution.task_id == task_id).all()

    @provide_session
    def get_latest_job_id(self, session):
        return session.query(BaseJob).order_by(sqlalchemy.desc(BaseJob.id)).first().id

    def start_scheduler(self, file_path):
        self.scheduler = EventBasedSchedulerJob(
            dag_directory=file_path,
            server_uri="localhost:{}".format(self.port),
            executor=LocalExecutor(3),
            max_runs=-1,
            refresh_dag_dir_interval=30
        )
        print("scheduler starting")
        self.scheduler.run()

    def wait_for_task_execution(self, dag_id, task_id, expected_num):
        result = False
        check_nums = 100
        while check_nums > 0:
            time.sleep(2)
            check_nums = check_nums - 1
            tes = self.get_task_execution(dag_id, task_id)
            if len(tes) == expected_num:
                result = True
                break
        self.assertTrue(result)

    def wait_for_task(self, dag_id, task_id, expected_state):
        result = False
        check_nums = 100
        while check_nums > 0:
            time.sleep(2)
            check_nums = check_nums - 1
            with create_session() as session:
                ti = session.query(TaskInstance).filter(
                    TaskInstance.dag_id == dag_id,
                    TaskInstance.task_id == task_id
                ).first()
            if ti and ti.state == expected_state:
                result = True
                break
        self.assertTrue(result)

    def test_notification(self):
        self.client.send_event(BaseEvent(key='a', value='b'))

    def run_a_task_function(self):
        while True:
            with create_session() as session:
                tes = session.query(TaskExecution).filter(TaskExecution.dag_id == 'single',
                                                          TaskExecution.task_id == 'task_1').all()
                if len(tes) > 0:
                    break
                else:
                    time.sleep(1)
        self.client.send_event(StopSchedulerEvent(job_id=0).to_event())

    def test_run_a_task(self):
        t = threading.Thread(target=self.run_a_task_function, args=())
        t.setDaemon(True)
        t.start()
        self.start_scheduler('../../dags/test_single_task_dag.py')
        tes: List[TaskExecution] = self.get_task_execution("single", "task_1")
        self.assertEqual(len(tes), 1)

    def run_event_task_function(self):
        client = NotificationClient(server_uri="localhost:{}".format(self.port),
                                    default_namespace="")
        while True:
            with create_session() as session:
                tes = session.query(TaskExecution).filter(TaskExecution.dag_id == 'event_dag',
                                                          TaskExecution.task_id == 'task_1').all()
                if len(tes) > 0:
                    time.sleep(5)
                    client.send_event(BaseEvent(key='start', value='', event_type='', namespace=''))
                    while True:
                        with create_session() as session_2:
                            tes_2 = session_2.query(TaskExecution).filter(TaskExecution.dag_id == 'event_dag',
                                                                          TaskExecution.task_id == 'task_2').all()
                            if len(tes_2) > 0:
                                break
                            else:
                                time.sleep(1)
                    break
                else:
                    time.sleep(1)
        client.send_event(StopSchedulerEvent(job_id=0).to_event())

    def test_run_event_task(self):
        t = threading.Thread(target=self.run_event_task_function, args=())
        t.setDaemon(True)
        t.start()
        self.start_scheduler('../../dags/test_event_task_dag.py')
        tes: List[TaskExecution] = self.get_task_execution("event_dag", "task_2")
        self.assertEqual(len(tes), 1)

    def run_trigger_dag_function(self):
        ns_client = NotificationClient(server_uri="localhost:{}".format(self.port), default_namespace="")
        client = EventSchedulerClient(ns_client=ns_client)
        while True:
            with create_session() as session:
                tes = session.query(TaskExecution).filter(TaskExecution.dag_id == 'trigger_dag',
                                                          TaskExecution.task_id == 'task_1').all()
                if len(tes) > 0:
                    break
                else:
                    client.trigger_parse_dag()
                    result = client.schedule_dag('trigger_dag')
                    print('result {}'.format(result.dagrun_id))
                time.sleep(5)
        ns_client.send_event(StopSchedulerEvent(job_id=0).to_event())

    def test_run_trigger_dag(self):
        import multiprocessing
        p = multiprocessing.Process(target=self.run_trigger_dag_function, args=())
        p.start()
        self.start_scheduler('../../dags/test_run_trigger_dag.py')
        tes: List[TaskExecution] = self.get_task_execution("trigger_dag", "task_1")
        self.assertEqual(len(tes), 1)

    def run_no_dag_file_function(self):
        ns_client = NotificationClient(server_uri="localhost:{}".format(self.port), default_namespace="")
        client = EventSchedulerClient(ns_client=ns_client)
        with create_session() as session:
            client.trigger_parse_dag()
            result = client.schedule_dag('no_dag')
            print('result {}'.format(result.dagrun_id))
            time.sleep(5)
        ns_client.send_event(StopSchedulerEvent(job_id=0).to_event())

    def test_no_dag_file_trigger_dag(self):
        import multiprocessing
        p = multiprocessing.Process(target=self.run_no_dag_file_function, args=())
        p.start()
        self.start_scheduler('../../dags/test_run_trigger_dag.py')
        tes: List[TaskExecution] = self.get_task_execution("trigger_dag", "task_1")
        self.assertEqual(len(tes), 0)

    def run_trigger_task_function(self):
        # waiting parsed dag file done,
        time.sleep(5)
        ns_client = NotificationClient(server_uri="localhost:{}".format(self.port), default_namespace="a")
        client = EventSchedulerClient(ns_client=ns_client)
        execution_context = client.schedule_dag('trigger_task')
        while True:
            with create_session() as session:
                tes = session.query(TaskExecution).filter(TaskExecution.dag_id == 'trigger_task',
                                                          TaskExecution.task_id == 'task_1').all()
                if len(tes) > 0:
                    client.schedule_task('task_2', SchedulingAction.START, execution_context)
                    while True:
                        with create_session() as session_2:
                            tes_2 = session_2.query(TaskExecution).filter(TaskExecution.dag_id == 'trigger_task',
                                                                          TaskExecution.task_id == 'task_2').all()
                            if len(tes_2) > 0:
                                break
                            else:
                                time.sleep(1)
                    break
                else:
                    time.sleep(1)
        ns_client.send_event(StopSchedulerEvent(job_id=0).to_event())

    def test_task_trigger_dag(self):
        import threading
        t = threading.Thread(target=self.run_trigger_task_function, args=())
        t.setDaemon(True)
        t.start()
        self.start_scheduler('../../dags/test_task_trigger_dag.py')
        tes: List[TaskExecution] = self.get_task_execution("trigger_task", "task_2")
        self.assertEqual(len(tes), 1)

    def run_ai_flow_function(self):
        client = NotificationClient(server_uri="localhost:{}".format(self.port),
                                    default_namespace="default")
        while True:
            with create_session() as session:
                tes = session.query(TaskExecution).filter(TaskExecution.dag_id == 'workflow_1',
                                                          TaskExecution.task_id == '0_job').all()
                if len(tes) > 0:
                    time.sleep(5)
                    client.send_event(BaseEvent(key='key_1', value='value_1', event_type='UNDEFINED'))
                    client.send_event(BaseEvent(key='key_2', value='value_2', event_type='UNDEFINED'))
                    while True:
                        with create_session() as session_2:
                            tes_2 = session_2.query(TaskExecution).filter(TaskExecution.dag_id == 'workflow_1').all()
                            if len(tes_2) == 3:
                                break
                            else:
                                time.sleep(1)
                    break
                else:
                    time.sleep(1)
        time.sleep(3)
        client.send_event(StopSchedulerEvent(job_id=0).to_event())

    def test_run_ai_flow_dag(self):
        import threading
        t = threading.Thread(target=self.run_ai_flow_function, args=())
        t.setDaemon(True)
        t.start()
        self.start_scheduler('../../dags/test_aiflow_dag.py')
        tes: List[TaskExecution] = self.get_task_execution("workflow_1", "1_job")
        self.assertEqual(len(tes), 1)

    def stop_dag_function(self):
        stopped = False
        while not stopped:
            tes = self.get_task_execution(EVENT_BASED_SCHEDULER_DAG, 'sleep_to_be_stopped')
            if tes and len(tes) == 1:
                self.client.send_event(StopDagEvent(EVENT_BASED_SCHEDULER_DAG).to_event())
                while not stopped:
                    tes2 = self.get_task_execution(EVENT_BASED_SCHEDULER_DAG, 'sleep_to_be_stopped')
                    if tes2[0].state == State.KILLED:
                        stopped = True
                        time.sleep(5)
                    else:
                        time.sleep(1)
            else:
                time.sleep(1)
        self.client.send_event(StopSchedulerEvent(job_id=0).to_event())

    def test_stop_dag(self):
        t = threading.Thread(target=self.stop_dag_function)
        t.setDaemon(True)
        t.start()
        self.start_scheduler('../../dags/test_event_based_scheduler.py')
        with create_session() as session:
            from airflow.models import DagModel
            dag_model: DagModel = DagModel.get_dagmodel(EVENT_BASED_SCHEDULER_DAG)
            self.assertTrue(dag_model.is_paused)
            self.assertEqual(dag_model.get_last_dagrun().state, "killed")
            for ti in session.query(TaskInstance).filter(TaskInstance.dag_id == EVENT_BASED_SCHEDULER_DAG):
                self.assertTrue(ti.state in [State.SUCCESS, State.KILLED])
            for te in session.query(TaskExecution).filter(TaskExecution.dag_id == EVENT_BASED_SCHEDULER_DAG):
                self.assertTrue(te.state in [State.SUCCESS, State.KILLED])

    # def test_vvp_dag(self):
    #     file_path = '/tmp/airflow/test_vvp_dag.py'
    #     # file_path = '../../dags/test_vvp_dag.py'
    #     scheduler = EventBasedSchedulerJob(
    #         dag_directory=file_path,
    #         server_uri="localhost:{}".format(self.port),
    #         max_runs=-1,
    #         refresh_dag_dir_interval=5
    #     )
    #     print("scheduler starting")
    #     scheduler.run()
