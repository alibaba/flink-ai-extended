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
from typing import Text
import traceback
from ai_flow.project.blob_manager import BlobManagerFactory

from ai_flow.common import json_utils
from ai_flow.project.project_description import ProjectDesc, get_project_description_from
from ai_flow.rest_endpoint.protobuf.message_pb2 import ResultProto, StatusProto
from ai_flow.rest_endpoint.protobuf.scheduling_service_pb2_grpc import SchedulingServiceServicer
from ai_flow.rest_endpoint.protobuf.scheduling_service_pb2 import \
    (ScheduleWorkflowRequest,
     WorkflowInfoResponse,
     ListWorkflowInfoResponse,
     WorkflowExecutionRequest,
     WorkflowExecutionResponse,
     ListWorkflowExecutionResponse,
     ScheduleJobRequest,
     JobInfoResponse,
     ListJobInfoResponse)
from ai_flow.scheduler.scheduler_factory import SchedulerFactory
from ai_flow.scheduler.scheduler_interface import AbstractScheduler, SchedulerConfig
from ai_flow.workflow.workflow import Workflow
from ai_flow.rest_endpoint.service.workflow_proto_utils import workflow_to_proto, workflow_list_to_proto, \
    workflow_execution_to_proto, workflow_execution_list_to_proto, job_to_proto, job_list_to_proto


class SchedulingService(SchedulingServiceServicer):
    def __init__(self,
                 scheduler_config: SchedulerConfig,
                 notification_service_uri: Text = None):
        self._notification_service_uri = notification_service_uri
        self._scheduler_config = scheduler_config
        self._scheduler: AbstractScheduler = SchedulerFactory.create_scheduler(scheduler_config)

    # workflow interface

    def submitWorkflow(self, request, context):
        try:
            rq: ScheduleWorkflowRequest = request
            workflow: Workflow = json_utils.loads(rq.workflow_json)
            workflow.workflow_name = rq.workflow_name
            config = {}
            config.update(workflow.project_desc.project_config['blob'])
            # config['local_repository'] = self._scheduler_config.repository()
            blob_manager = BlobManagerFactory.get_blob_manager(config)
            project_path: Text = blob_manager\
                .download_blob(workflow_id=workflow.workflow_id,
                               remote_path=workflow.project_desc.project_config.get('uploaded_project_path'),
                               local_path=self._scheduler_config.repository())

            project_desc: ProjectDesc = get_project_description_from(project_path)
            project_name = project_desc.project_name
            # update workflow
            workflow.project_desc = project_desc
            for n, j in workflow.jobs.items():
                j.job_config.project_desc = project_desc
                j.job_config.project_path = project_path

            workflow_info = self._scheduler.submit_workflow(workflow, project_desc)
            if workflow_info is None:
                return WorkflowInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{}, {} do not exist!'.format(project_name,
                                                                                   workflow.workflow_name)))
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.OK),
                                        workflow=workflow_to_proto(workflow_info))
        except Exception as err:
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                           error_message=traceback.format_exc()))

    def deleteWorkflow(self, request, context):
        try:
            rq: ScheduleWorkflowRequest = request
            workflow_info = self._scheduler.delete_workflow(rq.namespace, rq.workflow_name)
            if workflow_info is None:
                return WorkflowInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{},{} do not exist!'.format(rq.namespace, rq.workflow_name)))
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.OK),
                                        workflow=workflow_to_proto(workflow_info))
        except Exception as err:
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                           error_message=traceback.format_exc()))

    def pauseWorkflowScheduling(self, request, context):
        try:
            rq: ScheduleWorkflowRequest = request
            workflow_info = self._scheduler.pause_workflow_scheduling(rq.namespace, rq.workflow_name)
            if workflow_info is None:
                return WorkflowInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{} do not exist!'.format(rq.workflow_name)))
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.OK),
                                        workflow=workflow_to_proto(workflow_info))
        except Exception as err:
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                           error_message=traceback.format_exc()))

    def resumeWorkflowScheduling(self, request, context):
        try:
            rq: ScheduleWorkflowRequest = request
            workflow_info = self._scheduler.resume_workflow_scheduling(rq.namespace, rq.workflow_name)
            if workflow_info is None:
                return WorkflowInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{} do not exist!'.format(rq.workflow_name)))
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.OK),
                                        workflow=workflow_to_proto(workflow_info))
        except Exception as err:
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                           error_message=traceback.format_exc()))

    def getWorkflow(self, request, context):
        try:
            rq: ScheduleWorkflowRequest = request
            workflow_info = self._scheduler.get_workflow(rq.namespace, rq.workflow_name)
            if workflow_info is None:
                return WorkflowInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{} do not exist!'.format(rq.workflow_name)))
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.OK),
                                        workflow=workflow_to_proto(workflow_info))
        except Exception as err:
            return WorkflowInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                           error_message=traceback.format_exc()))

    def listWorkflows(self, request, context):
        try:
            rq: ScheduleWorkflowRequest = request
            workflow_info_list = self._scheduler.list_workflows(rq.namespace)
            workflow_proto_list = workflow_list_to_proto(workflow_info_list)
            response = ListWorkflowInfoResponse(result=ResultProto(status=StatusProto.OK))
            response.workflow_list.extend(workflow_proto_list)
            return response
        except Exception as err:
            return ListWorkflowInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                               error_message=traceback.format_exc()))

    # workflow execution interface

    def startNewWorkflowExecution(self, request, context):
        try:
            rq: WorkflowExecutionRequest = request
            workflow_execution = self._scheduler.start_new_workflow_execution(rq.namespace, rq.workflow_name)
            if workflow_execution is None:
                return WorkflowInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{}, {} do not exist!'.format(rq.namespace, rq.workflow_name)))
            return WorkflowExecutionResponse(result=ResultProto(status=StatusProto.OK),
                                             workflow_execution=workflow_execution_to_proto(workflow_execution))
        except Exception as err:
            return WorkflowExecutionResponse(result=ResultProto(status=StatusProto.ERROR,
                                                                error_message=traceback.format_exc()))

    def killAllWorkflowExecutions(self, request, context):
        try:
            rq: WorkflowExecutionRequest = request
            workflow_execution_list = self._scheduler.kill_all_workflow_execution(rq.namespace, rq.workflow_name)
            response = ListWorkflowExecutionResponse(result=ResultProto(status=StatusProto.OK))
            response.workflow_execution_list.extend(workflow_execution_list_to_proto(workflow_execution_list))
            return response
        except Exception as err:
            return ListWorkflowExecutionResponse(result=ResultProto(status=StatusProto.ERROR,
                                                                    error_message=traceback.format_exc()))

    def killWorkflowExecution(self, request, context):
        try:
            rq: WorkflowExecutionRequest = request
            workflow_execution = self._scheduler.kill_workflow_execution(rq.execution_id)
            if workflow_execution is None:
                return WorkflowExecutionResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{} do not exist!'.format(rq.execution_id)))
            return WorkflowExecutionResponse(result=ResultProto(status=StatusProto.OK),
                                             workflow_execution=workflow_execution_to_proto(workflow_execution))
        except Exception as err:
            return WorkflowExecutionResponse(result=ResultProto(status=StatusProto.ERROR,
                                                                error_message=traceback.format_exc()))

    def getWorkflowExecution(self, request, context):
        try:
            rq: WorkflowExecutionRequest = request
            workflow_execution = self._scheduler.get_workflow_execution(rq.execution_id)
            if workflow_execution is None:
                return WorkflowExecutionResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{} do not exist!'.format(rq.execution_id)))
            return WorkflowExecutionResponse(result=ResultProto(status=StatusProto.OK),
                                             workflow_execution=workflow_execution_to_proto(workflow_execution))
        except Exception as err:
            return WorkflowExecutionResponse(result=ResultProto(status=StatusProto.ERROR,
                                                                error_message=traceback.format_exc()))

    def listWorkflowExecutions(self, request, context):
        try:
            rq: WorkflowExecutionRequest = request
            workflow_execution_list = self._scheduler.list_workflow_executions(rq.namespace, rq.workflow_name)
            response = ListWorkflowExecutionResponse(result=ResultProto(status=StatusProto.OK))
            response.workflow_execution_list.extend(workflow_execution_list_to_proto(workflow_execution_list))
            return response
        except Exception as err:
            return ListWorkflowExecutionResponse(result=ResultProto(status=StatusProto.ERROR,
                                                                    error_message=traceback.format_exc()))

    # job interface
    def startJob(self, request, context):
        try:
            rq: ScheduleJobRequest = request
            job = self._scheduler.start_job(rq.job_name, rq.execution_id)
            if job is None:
                return JobInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{} do not exist!'.format(rq.job_name)))
            return JobInfoResponse(result=ResultProto(status=StatusProto.OK), job=job_to_proto(job))
        except Exception as err:
            return JobInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                      error_message=traceback.format_exc()))

    def stopJob(self, request, context):
        try:
            rq: ScheduleJobRequest = request
            job = self._scheduler.stop_job(rq.job_name, rq.execution_id)
            if job is None:
                return JobInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{} do not exist!'.format(rq.job_name)))
            return JobInfoResponse(result=ResultProto(status=StatusProto.OK), job=job_to_proto(job))
        except Exception as err:
            return JobInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                      error_message=traceback.format_exc()))

    def restartJob(self, request, context):
        try:
            rq: ScheduleJobRequest = request
            job = self._scheduler.restart_job(rq.job_name, rq.execution_id)
            if job is None:
                return JobInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{} do not exist!'.format(rq.job_name)))
            return JobInfoResponse(result=ResultProto(status=StatusProto.OK), job=job_to_proto(job))
        except Exception as err:
            return JobInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                      error_message=traceback.format_exc()))

    def getJob(self, request, context):
        try:
            rq: ScheduleJobRequest = request
            job = self._scheduler.get_job(rq.job_name, rq.execution_id)
            if job is None:
                return JobInfoResponse(
                    result=ResultProto(status=StatusProto.ERROR,
                                       error_message='{} do not exist!'.format(rq.job_name)))
            return JobInfoResponse(result=ResultProto(status=StatusProto.OK), job=job_to_proto(job))
        except Exception as err:
            return JobInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                      error_message=traceback.format_exc()))

    def listJobs(self, request, context):
        try:
            rq: ScheduleJobRequest = request
            job_list = self._scheduler.list_jobs(rq.execution_id)
            response = ListJobInfoResponse(result=ResultProto(status=StatusProto.OK))
            response.job_list.extend(job_list_to_proto(job_list))
            return response
        except Exception as err:
            return ListJobInfoResponse(result=ResultProto(status=StatusProto.ERROR,
                                                          error_message=traceback.format_exc()))
