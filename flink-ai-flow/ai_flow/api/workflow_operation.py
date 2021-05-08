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
import time
import sys
import os
from typing import Text, List, Dict
from ai_flow.project.blob_manager import BlobManagerFactory
from ai_flow.common import json_utils
from ai_flow.graph.graph import default_graph
from ai_flow.translator.base_translator import get_default_translator
from ai_flow.client.ai_flow_client import get_ai_flow_client
from ai_flow.api.configuration import project_config, project_description
from ai_flow.workflow.workflow import JobInfo, WorkflowExecutionInfo, WorkflowInfo, Workflow
from ai_flow.rest_endpoint.service.workflow_proto_utils import \
    proto_to_workflow, proto_to_workflow_list, proto_to_workflow_execution, proto_to_workflow_execution_list,\
    proto_to_job, proto_to_job_list


def _upload_project_package(workflow: Workflow):
    """
    Upload the project package.

    :param workflow: The generated workflow.
    """
    project_desc = project_description()
    with open(project_desc.get_absolute_temp_path() + "/"
              + project_desc.project_config.get_project_uuid() + "_workflow.json", 'w') as f:
        f.write(json_utils.dumps(workflow))
    blob_manager = BlobManagerFactory.get_blob_manager(project_desc.project_config['blob'])
    uploaded_project_path = blob_manager.upload_blob(str(workflow.workflow_id), project_desc.project_path)
    project_desc.project_config.set_uploaded_project_path(uploaded_project_path)
    for job in workflow.jobs.values():
        job.job_config.project_path = uploaded_project_path


def _register_job_meta(workflow_id: int, job):

    start_time = time.time()
    if job.job_config.job_name is None:
        name = job.instance_id
    else:
        name = job.job_config.job_name
    job_name = str(workflow_id) + '_' + name[0:20] + '_' + str(start_time)
    job.job_name = job_name


def _set_entry_module_path(workflow: Workflow, entry_module_path: Text):
    """
    Set entry model path.
    :param workflow: The generated workflow.
    """
    for job in workflow.jobs.values():
        job.job_config.properties['entry_module_path'] = entry_module_path


def submit_workflow(workflow_name: Text = None,
                    args: Dict = None) -> WorkflowInfo:
    """
    Submit the ai flow workflow to the scheduler.
    :param workflow_name: The ai flow workflow identify.
    :param args: The arguments of the submit action.
    :return: The result of the submit action.
    """
    call_path = os.path.abspath(sys._getframe(1).f_code.co_filename)
    project_path = os.path.abspath(project_description().project_path)
    entry_module_path = call_path[len(project_path)+14:-3].replace('/', '.')
    namespace = project_config().get_project_name()
    translator = get_default_translator()
    workflow = translator.translate(graph=default_graph(), project_desc=project_description())
    for job in workflow.jobs.values():
        _register_job_meta(workflow_id=workflow.workflow_id, job=job)
    _set_entry_module_path(workflow, entry_module_path)
    _upload_project_package(workflow)
    return proto_to_workflow(get_ai_flow_client()
                             .submit_workflow_to_scheduler(namespace=namespace,
                                                           workflow_json=json_utils.dumps(workflow),
                                                           workflow_name=workflow_name,
                                                           args=args))


def delete_workflow(workflow_name: Text = None) -> WorkflowInfo:
    """
    Delete the ai flow workflow from the scheduler.
    :param workflow_name: The ai flow workflow identify.
    :return: The result of the action.
    """
    namespace = project_config().get_project_name()
    return proto_to_workflow(get_ai_flow_client().delete_workflow(namespace, workflow_name))


def pause_workflow_scheduling(workflow_name: Text = None) -> WorkflowInfo:
    """
    Pause the ai flow workflow from the scheduler.
    :param workflow_name: The ai flow workflow identify.
    :return: The result of the action.
    """
    namespace = project_config().get_project_name()
    return proto_to_workflow(get_ai_flow_client().pause_workflow_scheduling(namespace, workflow_name))


def resume_workflow_scheduling(workflow_name: Text = None) -> WorkflowInfo:
    """
    Resume the ai flow workflow from the scheduler.
    :param workflow_name: The ai flow workflow identify.
    :return: The result of the action.
    """
    namespace = project_config().get_project_name()
    return proto_to_workflow(get_ai_flow_client().resume_workflow_scheduling(namespace, workflow_name))


def get_workflow(workflow_name: Text = None) -> WorkflowInfo:
    """
    Return the workflow information.
    :param workflow_name: The ai flow workflow identify.
    :return: the workflow information.
    """
    namespace = project_config().get_project_name()
    return proto_to_workflow(get_ai_flow_client().get_workflow(namespace, workflow_name))


def list_workflows() -> List[WorkflowInfo]:
    """
    :return: All workflow information.
    """
    namespace = project_config().get_project_name()
    return proto_to_workflow_list(get_ai_flow_client().list_workflows(namespace))


def start_new_workflow_execution(workflow_name: Text) -> WorkflowExecutionInfo:
    """
    Run the project under the current project path.
    :param workflow_name: The ai flow workflow identify.
    :return: The result of the run action.
    """
    namespace = project_config().get_project_name()
    return proto_to_workflow_execution(get_ai_flow_client().start_new_workflow_execution(namespace, workflow_name))


def kill_all_workflow_executions(workflow_name: Text) -> List[WorkflowExecutionInfo]:
    """
    Stop all instances of the workflow.
    :param workflow_name: The ai flow workflow identify.
    :return: The result of the action.
    """
    namespace = project_config().get_project_name()
    return proto_to_workflow_execution_list(get_ai_flow_client().kill_all_workflow_executions(namespace, workflow_name))


def kill_workflow_execution(execution_id: Text) -> WorkflowExecutionInfo:
    """
    Stop the instance of the workflow.
    :param execution_id: The ai flow workflow execution identify.
    :return: The result of the action.
    """
    return proto_to_workflow_execution(get_ai_flow_client().kill_workflow_execution(execution_id))


def get_workflow_execution(execution_id: Text) -> WorkflowExecutionInfo:
    """
    Get the WorkflowExecutionInfo from scheduler.
    :param execution_id:
    :return: WorkflowExecutionInfo
    """
    return proto_to_workflow_execution(get_ai_flow_client().get_workflow_execution(execution_id))


def list_workflow_executions(workflow_name: Text) -> List[WorkflowExecutionInfo]:
    """
    :param workflow_name: The ai flow workflow identify.
    :return: All workflow executions of the workflow.
    """
    namespace = project_config().get_project_name()
    return proto_to_workflow_execution_list(get_ai_flow_client().list_workflow_executions(namespace, workflow_name))


def start_job(job_name: Text,
              execution_id: Text) -> JobInfo:
    """
    Start a job defined in the ai flow workflow.
    :param job_name: The job name which task defined in workflow.
    :param execution_id: The ai flow workflow execution identify.
    :return: The result of the action.
    """
    return proto_to_job(get_ai_flow_client().start_job(job_name, execution_id))


def stop_job(job_name: Text,
             execution_id: Text) -> JobInfo:
    """
    Stop a job defined in the ai flow workflow.
    :param job_name: The job name which task defined in workflow.
    :param execution_id: The ai flow workflow execution identify.
    :return: The result of the action.
    """
    return proto_to_job(get_ai_flow_client().stop_job(job_name, execution_id))


def restart_job(job_name: Text,
                execution_id: Text) -> JobInfo:
    """
    Restart a task defined in the ai flow workflow.
    :param job_name: The job name which task defined in workflow.
    :param execution_id: The ai flow workflow execution identify.
    :return: The result of the action.
    """
    return proto_to_job(get_ai_flow_client().restart_job(job_name, execution_id))


def get_job(job_name: Text,
            execution_id: Text) -> JobInfo:
    """
    Get job information by job name.
    :param job_name:
    :param execution_id:
    :return:
    """
    return proto_to_job(get_ai_flow_client().get_job(job_name, execution_id))


def list_jobs(execution_id: Text) -> List[JobInfo]:
    """
    List the jobs of the workflow execution.
    :param execution_id:
    :return:
    """
    return proto_to_job_list(get_ai_flow_client().list_jobs(execution_id))
