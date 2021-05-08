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
import mock
import os
import unittest

from typing import Text, List, Dict

from ai_flow.meta.job_meta import State
from ai_flow.rest_endpoint.protobuf.message_pb2 import StateProto

from ai_flow.rest_endpoint.service.client.scheduling_client import SchedulingClient
from ai_flow.rest_endpoint.service.server import AIFlowServer
from ai_flow.project.project_description import ProjectDesc
from ai_flow.scheduler.scheduler_interface import AbstractScheduler, SchedulerConfig
from ai_flow.workflow.workflow import JobInfo, WorkflowExecutionInfo, WorkflowInfo, Workflow

_SQLITE_DB_FILE = 'aiflow.db'
_SQLITE_DB_URI = '%s%s' % ('sqlite:///', _SQLITE_DB_FILE)
_PORT = '50051'


class MockScheduler(AbstractScheduler):
    def submit_workflow(self, workflow: Workflow, project_desc: ProjectDesc, args: Dict = None) -> WorkflowInfo:
        pass

    def delete_workflow(self, project_name: Text, workflow_name: Text) -> WorkflowInfo:
        pass

    def pause_workflow_scheduling(self, project_name: Text, workflow_name: Text) -> WorkflowInfo:
        pass

    def resume_workflow_scheduling(self, project_name: Text, workflow_name: Text) -> WorkflowInfo:
        pass

    def get_workflow(self, project_name: Text, workflow_name: Text) -> WorkflowInfo:
        pass

    def list_workflows(self, project_name: Text) -> List[WorkflowInfo]:
        pass

    def start_new_workflow_execution(self, project_name: Text, workflow_name: Text) -> WorkflowExecutionInfo:
        pass

    def kill_all_workflow_execution(self, project_name: Text, workflow_name: Text) -> List[WorkflowExecutionInfo]:
        pass

    def kill_workflow_execution(self, execution_id: Text) -> WorkflowExecutionInfo:
        pass

    def get_workflow_execution(self, execution_id: Text) -> WorkflowExecutionInfo:
        pass

    def list_workflow_executions(self, project_name: Text, workflow_name: Text) -> List[WorkflowExecutionInfo]:
        pass

    def start_job(self, job_name: Text, execution_id: Text) -> JobInfo:
        pass

    def stop_job(self, job_name: Text, execution_id: Text) -> JobInfo:
        pass

    def restart_job(self, job_name: Text, execution_id: Text) -> JobInfo:
        pass

    def get_job(self, job_name: Text, execution_id: Text) -> JobInfo:
        pass

    def list_jobs(self, execution_id: Text) -> List[JobInfo]:
        pass


class TestSchedulingService(unittest.TestCase):
    def setUp(self):
        config = SchedulerConfig()
        config.set_scheduler_class_name('ai_flow.test.scheduler.test_scheduling_service.MockScheduler')
        if os.path.exists(_SQLITE_DB_FILE):
            os.remove(_SQLITE_DB_FILE)
        self.server = AIFlowServer(store_uri=_SQLITE_DB_URI, port=_PORT,
                                   start_default_notification=False,
                                   start_deploy_service=False,
                                   start_meta_service=False,
                                   start_metric_service=False,
                                   start_model_center_service=False,
                                   start_scheduling_service=True,
                                   scheduler_config=config)
        self.server.run()

    def tearDown(self):
        self.server.stop()
        if os.path.exists(_SQLITE_DB_FILE):
            os.remove(_SQLITE_DB_FILE)

    # def test_submit_workflow(self):
    #     with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
    #         instance = mockScheduler.return_value
    #         instance.submit_workflow.return_value = WorkflowInfo(workflow_name='test_workflow')
    #         client = SchedulingClient("localhost:{}".format(_PORT))
    #         workflow = client.submit_workflow_to_scheduler(namespace='namespace', workflow_name='test_workflow')
    #         print(workflow)

    def test_delete_none_workflow(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.delete_workflow.return_value = None
            client = SchedulingClient("localhost:{}".format(_PORT))
            with self.assertRaises(Exception) as context:
                workflow = client.delete_workflow(namespace='namespace', workflow_name='test_workflow')

    def test_delete_workflow(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.delete_workflow.return_value = WorkflowInfo(workflow_name='test_workflow')
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow = client.delete_workflow(namespace='namespace', workflow_name='test_workflow')
            self.assertTrue('test_workflow', workflow.name)

    def test_pause_workflow(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.pause_workflow_scheduling.return_value = WorkflowInfo(workflow_name='test_workflow')
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow = client.pause_workflow_scheduling(namespace='namespace', workflow_name='test_workflow')
            self.assertTrue('test_workflow', workflow.name)

    def test_resume_workflow(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.resume_workflow_scheduling.return_value = WorkflowInfo(workflow_name='test_workflow')
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow = client.resume_workflow_scheduling(namespace='namespace', workflow_name='test_workflow')
            self.assertTrue('test_workflow', workflow.name)

    def test_get_workflow(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.get_workflow.return_value = WorkflowInfo(workflow_name='test_workflow')
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow = client.get_workflow(namespace='namespace', workflow_name='test_workflow')
            self.assertTrue('test_workflow', workflow.name)

    def test_list_workflows(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.list_workflows.return_value = [WorkflowInfo(workflow_name='test_workflow_1'),
                                                    WorkflowInfo(workflow_name='test_workflow_2')]
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow_list = client.list_workflows(namespace='namespace')
            self.assertTrue(2, len(workflow_list))

    def test_start_new_workflow_execution(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.start_new_workflow_execution.return_value \
                = WorkflowExecutionInfo(execution_id='id', state=State.INIT)
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow_execution = client.start_new_workflow_execution(namespace='namespace',
                                                                     workflow_name='test_workflow')
            self.assertEqual('id', workflow_execution.execution_id)
            self.assertEqual(StateProto.INIT, workflow_execution.execution_state)

    def test_kill_all_workflow_execution(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.kill_all_workflow_execution.return_value \
                = [WorkflowExecutionInfo(execution_id='id_1', state=State.INIT),
                   WorkflowExecutionInfo(execution_id='id_2', state=State.INIT)]
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow_execution_list = client.kill_all_workflow_executions(namespace='namespace',
                                                                          workflow_name='test_workflow')
            self.assertEqual(2, len(workflow_execution_list))

    def test_kill_workflow_execution(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.kill_workflow_execution.return_value \
                = WorkflowExecutionInfo(execution_id='id', state=State.RUNNING)
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow_execution = client.kill_workflow_execution(execution_id='id')
            self.assertEqual('id', workflow_execution.execution_id)
            self.assertEqual(StateProto.RUNNING, workflow_execution.execution_state)

    def test_get_workflow_execution(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.get_workflow_execution.return_value \
                = WorkflowExecutionInfo(execution_id='id', state=State.INIT)
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow_execution = client.get_workflow_execution(execution_id='id')
            self.assertEqual('id', workflow_execution.execution_id)
            self.assertEqual(StateProto.INIT, workflow_execution.execution_state)

    def test_list_workflow_executions(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.list_workflow_executions.return_value \
                = [WorkflowExecutionInfo(execution_id='id_1', state=State.INIT),
                   WorkflowExecutionInfo(execution_id='id_2', state=State.INIT)]
            client = SchedulingClient("localhost:{}".format(_PORT))
            workflow_execution_list = client.list_workflow_executions(namespace='namespace',
                                                                      workflow_name='test_workflow')
            self.assertEqual(2, len(workflow_execution_list))

    def test_start_job(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.start_job.return_value \
                = JobInfo(job_name='job_name',
                          state=State.RUNNING,
                          workflow_execution=WorkflowExecutionInfo(execution_id='id', state=State.INIT))
            client = SchedulingClient("localhost:{}".format(_PORT))
            job = client.start_job(job_name='job_name', execution_id='id')
            self.assertEqual('job_name', job.name)
            self.assertEqual(StateProto.RUNNING, job.job_state)
            self.assertEqual('id', job.workflow_execution.execution_id)
            self.assertEqual(StateProto.INIT, job.workflow_execution.execution_state)

    def test_stop_job(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.stop_job.return_value \
                = JobInfo(job_name='job_name',
                          state=State.RUNNING,
                          workflow_execution=WorkflowExecutionInfo(execution_id='id', state=State.INIT))
            client = SchedulingClient("localhost:{}".format(_PORT))
            job = client.stop_job(job_name='job_name', execution_id='id')
            self.assertEqual('job_name', job.name)
            self.assertEqual(StateProto.RUNNING, job.job_state)
            self.assertEqual('id', job.workflow_execution.execution_id)
            self.assertEqual(StateProto.INIT, job.workflow_execution.execution_state)

    def test_restart_job(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.restart_job.return_value \
                = JobInfo(job_name='job_name',
                          state=State.RUNNING,
                          workflow_execution=WorkflowExecutionInfo(execution_id='id', state=State.INIT))
            client = SchedulingClient("localhost:{}".format(_PORT))
            job = client.restart_job(job_name='job_name', execution_id='id')
            self.assertEqual('job_name', job.name)
            self.assertEqual(StateProto.RUNNING, job.job_state)
            self.assertEqual('id', job.workflow_execution.execution_id)
            self.assertEqual(StateProto.INIT, job.workflow_execution.execution_state)

    def test_get_job(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.get_job.return_value \
                = JobInfo(job_name='job_name',
                          state=State.RUNNING,
                          workflow_execution=WorkflowExecutionInfo(execution_id='id', state=State.INIT))
            client = SchedulingClient("localhost:{}".format(_PORT))
            job = client.get_job(job_name='job_name', execution_id='id')
            self.assertEqual('job_name', job.name)
            self.assertEqual(StateProto.RUNNING, job.job_state)
            self.assertEqual('id', job.workflow_execution.execution_id)
            self.assertEqual(StateProto.INIT, job.workflow_execution.execution_state)

    def test_list_jobs(self):
        with mock.patch('ai_flow.test.scheduler.test_scheduling_service.MockScheduler') as mockScheduler:
            instance = mockScheduler.return_value
            self.server.scheduling_service._scheduler = instance

            instance.list_jobs.return_value \
                = [JobInfo(job_name='job_name_1',
                           state=State.RUNNING,
                           workflow_execution=WorkflowExecutionInfo(execution_id='id', state=State.INIT)),
                   JobInfo(job_name='job_name_2',
                           state=State.RUNNING,
                           workflow_execution=WorkflowExecutionInfo(execution_id='id', state=State.INIT))]
            client = SchedulingClient("localhost:{}".format(_PORT))
            job_list = client.list_jobs(execution_id='id')
            self.assertEqual(2, len(job_list))
