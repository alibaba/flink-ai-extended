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
from typing import Text, List, Dict

from ai_flow.project.project_description import ProjectDesc
from ai_flow.scheduler.scheduler_factory import SchedulerFactory
from ai_flow.scheduler.scheduler_interface import AbstractScheduler, SchedulerConfig
from ai_flow.workflow.workflow import JobInfo, WorkflowExecutionInfo, WorkflowInfo, Workflow


class UnitTestScheduler(AbstractScheduler):
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


class TestSchedulerFactory(unittest.TestCase):

    def test_create_scheduler(self):
        print(AbstractScheduler.__class__.__name__, AbstractScheduler.__class__.__module__)
        config = SchedulerConfig()
        config.set_scheduler_class_name('ai_flow.test.scheduler.test_scheduler_factory.UnitTestScheduler')
        sc = SchedulerFactory.create_scheduler(config)
        self.assertTrue(isinstance(sc, AbstractScheduler))
