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
from typing import Text, List, Dict

import grpc
from ai_flow.rest_endpoint.protobuf import scheduling_service_pb2_grpc
from ai_flow.rest_endpoint.protobuf import scheduling_service_pb2
from ai_flow.rest_endpoint.protobuf.message_pb2 import WorkflowProto, WorkflowExecutionProto, JobProto, StatusProto

from ai_flow.rest_endpoint.service.client.base_client import BaseClient


class SchedulingClient(BaseClient):
    def __init__(self, server_uri):
        super(SchedulingClient, self).__init__(server_uri)
        channel = grpc.insecure_channel(server_uri)
        self.scheduling_stub = scheduling_service_pb2_grpc.SchedulingServiceStub(channel)

    def submit_workflow_to_scheduler(self,
                                     namespace: Text,
                                     workflow_json: Text,
                                     workflow_name: Text = None,
                                     args: Dict = None) -> WorkflowProto:
        """
        Submit the ai flow workflow to the scheduler.
        :param workflow_json:
        :param namespace:
        :param workflow_name: The ai flow workflow identify.
        :param args: The arguments of the submit action.
        :return: The result of the submit action.
        """
        request = scheduling_service_pb2.ScheduleWorkflowRequest()
        request.namespace = namespace
        request.workflow_name = workflow_name
        request.workflow_json = workflow_json
        if args is not None:
            for k, v in args.items():
                request.args[k] = v
        response = self.scheduling_stub.submitWorkflow(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow

    def delete_workflow(self,
                        namespace: Text,
                        workflow_name: Text = None) -> WorkflowProto:
        """
        Delete the ai flow workflow from the scheduler.
        :param namespace:
        :param workflow_name: The ai flow workflow identify.
        :return: The result of the action.
        """
        request = scheduling_service_pb2.ScheduleWorkflowRequest()
        request.namespace = namespace
        request.workflow_name = workflow_name
        response = self.scheduling_stub.deleteWorkflow(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow

    def pause_workflow_scheduling(self,
                                  namespace: Text,
                                  workflow_name: Text = None) -> WorkflowProto:
        """
        Pause the ai flow workflow from the scheduler.
        :param namespace:
        :param workflow_name: The ai flow workflow identify.
        :return: The result of the action.
        """
        request = scheduling_service_pb2.ScheduleWorkflowRequest()
        request.namespace = namespace
        request.workflow_name = workflow_name
        response = self.scheduling_stub.pauseWorkflowScheduling(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow

    def resume_workflow_scheduling(self,
                                   namespace: Text,
                                   workflow_name: Text = None) -> WorkflowProto:
        """
        Resume the ai flow workflow from the scheduler.
        :param namespace:
        :param workflow_name: The ai flow workflow identify.
        :return: The result of the action.
        """
        request = scheduling_service_pb2.ScheduleWorkflowRequest()
        request.namespace = namespace
        request.workflow_name = workflow_name
        response = self.scheduling_stub.resumeWorkflowScheduling(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow

    def get_workflow(self,
                     namespace: Text,
                     workflow_name: Text = None) -> WorkflowProto:
        """
        Return the workflow information.
        :param namespace:
        :param workflow_name: The ai flow workflow identify.
        :return: the workflow information.
        """
        request = scheduling_service_pb2.ScheduleWorkflowRequest()
        request.namespace = namespace
        request.workflow_name = workflow_name
        response = self.scheduling_stub.getWorkflow(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow

    def list_workflows(self, namespace: Text) -> List[WorkflowProto]:
        """
        :return: All workflow information.
        """
        request = scheduling_service_pb2.ScheduleWorkflowRequest()
        request.namespace = namespace
        response = self.scheduling_stub.listWorkflows(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow_list

    def start_new_workflow_execution(self,
                                     namespace: Text,
                                     workflow_name: Text) -> WorkflowExecutionProto:
        """
        Run the project under the current project path.
        :param namespace:
        :param workflow_name: The ai flow workflow identify.
        :return: The result of the run action.
        """
        request = scheduling_service_pb2.WorkflowExecutionRequest()
        request.namespace = namespace
        request.workflow_name = workflow_name
        response = self.scheduling_stub.startNewWorkflowExecution(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow_execution

    def kill_all_workflow_executions(self,
                                     namespace: Text,
                                     workflow_name: Text) -> List[WorkflowExecutionProto]:
        """
        Stop all instances of the workflow.
        :param namespace:
        :param workflow_name: The ai flow workflow identify.
        :return: The result of the action.
        """
        request = scheduling_service_pb2.WorkflowExecutionRequest()
        request.namespace = namespace
        request.workflow_name = workflow_name
        response = self.scheduling_stub.killAllWorkflowExecutions(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow_execution_list

    def kill_workflow_execution(self,
                                execution_id: Text) -> WorkflowExecutionProto:
        """
        Stop the instance of the workflow.
        :param execution_id: The ai flow workflow execution identify.
        :return: The result of the action.
        """
        request = scheduling_service_pb2.WorkflowExecutionRequest()
        request.execution_id = execution_id
        response = self.scheduling_stub.killWorkflowExecution(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow_execution

    def get_workflow_execution(self,
                               execution_id: Text) -> WorkflowExecutionProto:
        """
        Get the WorkflowExecutionInfo from scheduler.
        :param execution_id:
        :return: WorkflowExecutionInfo
        """
        request = scheduling_service_pb2.WorkflowExecutionRequest()
        request.execution_id = execution_id
        response = self.scheduling_stub.getWorkflowExecution(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow_execution

    def list_workflow_executions(self,
                                 namespace: Text,
                                 workflow_name: Text) -> List[WorkflowExecutionProto]:
        """
        :param namespace:
        :param workflow_name: The ai flow workflow identify.
        :return: All workflow executions of the workflow.
        """
        request = scheduling_service_pb2.WorkflowExecutionRequest()
        request.namespace = namespace
        request.workflow_name = workflow_name
        response = self.scheduling_stub.listWorkflowExecutions(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.workflow_execution_list

    def start_job(self, job_name: Text,
                  execution_id: Text) -> JobProto:
        """
        Start a job defined in the ai flow workflow.
        :param job_name: The job name which task defined in workflow.
        :param execution_id: The ai flow workflow execution identify.
        :return: The result of the action.
        """
        request = scheduling_service_pb2.ScheduleJobRequest()
        request.job_name = job_name
        request.execution_id = execution_id
        response = self.scheduling_stub.startJob(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.job

    def stop_job(self, job_name: Text,
                 execution_id: Text) -> JobProto:
        """
        Stop a job defined in the ai flow workflow.
        :param job_name: The job name which task defined in workflow.
        :param execution_id: The ai flow workflow execution identify.
        :return: The result of the action.
        """
        request = scheduling_service_pb2.ScheduleJobRequest()
        request.job_name = job_name
        request.execution_id = execution_id
        response = self.scheduling_stub.stopJob(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.job

    def restart_job(self, job_name: Text,
                    execution_id: Text) -> JobProto:
        """
        Restart a task defined in the ai flow workflow.
        :param job_name: The job name which task defined in workflow.
        :param execution_id: The ai flow workflow execution identify.
        :return: The result of the action.
        """
        request = scheduling_service_pb2.ScheduleJobRequest()
        request.job_name = job_name
        request.execution_id = execution_id
        response = self.scheduling_stub.restartJob(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.job

    def get_job(self, job_name: Text,
                execution_id: Text) -> JobProto:
        """
        Get job information by job name.
        :param job_name:
        :param execution_id:
        :return:
        """
        request = scheduling_service_pb2.ScheduleJobRequest()
        request.job_name = job_name
        request.execution_id = execution_id
        response = self.scheduling_stub.getJob(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.job

    def list_jobs(self, execution_id: Text) -> List[JobProto]:
        """
        List the jobs of the workflow execution.
        :param execution_id:
        :return:
        """
        request = scheduling_service_pb2.ScheduleJobRequest()
        request.execution_id = execution_id
        response = self.scheduling_stub.listJobs(request)
        if response.result.status != StatusProto.OK:
            raise Exception(response.result.error_message)
        return response.job_list

