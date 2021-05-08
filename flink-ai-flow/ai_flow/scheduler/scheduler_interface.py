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
from abc import ABC, abstractmethod
from typing import Dict, Text, List, Optional

from ai_flow.common.configuration import AIFlowConfiguration
from ai_flow.workflow.workflow import Workflow, WorkflowInfo, WorkflowExecutionInfo, JobInfo
from ai_flow.project.project_description import ProjectDesc


class SchedulerConfig(AIFlowConfiguration):

    def repository(self):
        return self['repository']

    def set_repository(self, value):
        self['repository'] = value

    def scheduler_class_name(self):
        if self.get('scheduler_class_name') is not None:
            return self.get('scheduler_class_name')
        else:
            return 'ai_flow.scheduler.implements.airflow_scheduler.AirFlowScheduler'

    def set_scheduler_class_name(self, value):
        self['scheduler_class_name'] = value

    def notification_service_uri(self):
        return self.get('notification_service_uri', None)

    def set_notification_service_uri(self, value):
        self['notification_service_uri'] = value

    def properties(self):
        return self['properties']

    def set_properties(self, value):
        self['properties'] = value


class AbstractScheduler(ABC):
    def __init__(self, config: SchedulerConfig):
        self._config = config

    @property
    def config(self):
        return self._config

    @abstractmethod
    def submit_workflow(self, workflow: Workflow, project_desc: ProjectDesc, args: Dict = None) -> WorkflowInfo:
        pass

    @abstractmethod
    def delete_workflow(self, project_name: Text, workflow_name: Text) -> Optional[WorkflowInfo]:
        pass

    @abstractmethod
    def pause_workflow_scheduling(self, project_name: Text, workflow_name: Text) -> WorkflowInfo:
        pass

    @abstractmethod
    def resume_workflow_scheduling(self, project_name: Text, workflow_name: Text) -> WorkflowInfo:
        pass

    @abstractmethod
    def get_workflow(self, project_name: Text, workflow_name: Text) -> Optional[WorkflowInfo]:
        pass

    @abstractmethod
    def list_workflows(self, project_name: Text) -> List[WorkflowInfo]:
        pass

    @abstractmethod
    def start_new_workflow_execution(self, project_name: Text, workflow_name: Text) -> Optional[WorkflowExecutionInfo]:
        pass

    @abstractmethod
    def kill_all_workflow_execution(self, project_name: Text, workflow_name: Text) -> List[WorkflowExecutionInfo]:
        pass

    @abstractmethod
    def kill_workflow_execution(self, execution_id: Text) -> Optional[WorkflowExecutionInfo]:
        pass

    @abstractmethod
    def get_workflow_execution(self, execution_id: Text) -> Optional[WorkflowExecutionInfo]:
        pass

    @abstractmethod
    def list_workflow_executions(self, project_name: Text, workflow_name: Text) -> List[WorkflowExecutionInfo]:
        pass

    @abstractmethod
    def start_job(self, job_name: Text, execution_id: Text) -> JobInfo:
        pass

    @abstractmethod
    def stop_job(self, job_name: Text, execution_id: Text) -> JobInfo:
        pass

    @abstractmethod
    def restart_job(self, job_name: Text, execution_id: Text) -> JobInfo:
        pass

    @abstractmethod
    def get_job(self, job_name: Text, execution_id: Text) -> Optional[JobInfo]:
        pass

    @abstractmethod
    def list_jobs(self, execution_id: Text) -> List[JobInfo]:
        pass
