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
from tempfile import NamedTemporaryFile
from typing import Dict, Text, List, Optional
from ai_flow.meta import job_meta
from ai_flow.airflow.dag_generator import DAGGenerator
from ai_flow.project.project_description import ProjectDesc
from ai_flow.scheduler.scheduler_interface import AbstractScheduler, SchedulerConfig
from ai_flow.workflow.workflow import Workflow, WorkflowInfo, JobInfo, WorkflowExecutionInfo

from airflow.executors.scheduling_action import SchedulingAction
from airflow.models.taskexecution import TaskExecution
from airflow.models.taskinstance import TaskInstance
from airflow.models.taskstate import TaskState
from airflow.models.dag import DagTag, DagModel
from airflow.models.dagcode import DagCode
from airflow.models.dagrun import DagRun
from airflow.models.serialized_dag import SerializedDagModel
from airflow.utils.db import create_session
from airflow.utils.state import State
from airflow.contrib.jobs.scheduler_client import EventSchedulerClient, SCHEDULER_NAMESPACE, ExecutionContext


class AirFlowScheduler(AbstractScheduler):

    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.dag_generator = DAGGenerator()
        self._airflow_client = None

    @classmethod
    def airflow_dag_id(cls, namespace, workflow_name):
        return '{}.{}'.format(namespace, workflow_name)

    @classmethod
    def dag_id_to_namespace_workflow(cls, dag_id: Text):
        tmp = dag_id.split('.')
        return tmp[0], tmp[1]

    @classmethod
    def parse_namespace_workflow_name(cls, dag_id: Text):
        index = dag_id.find('.')
        return dag_id[0: index - 1], dag_id[index + 1:]

    @classmethod
    def airflow_state_to_state(cls, state):
        if State.SUCCESS == state:
            return job_meta.State.FINISHED
        elif State.FAILED == state:
            return job_meta.State.FAILED
        elif State.RUNNING == state:
            return job_meta.State.RUNNING
        elif State.KILLING == state:
            return job_meta.State.KILLING
        elif State.KILLED == state or State.SHUTDOWN == state:
            return job_meta.State.KILLED
        else:
            return job_meta.State.INIT

    @classmethod
    def dag_exist(cls, dag_id):
        with create_session() as session:
            dag = session.query(DagModel).filter(DagModel.dag_id == dag_id).first()
            if dag is None:
                return False
            else:
                return True

    @classmethod
    def dagrun_exist(cls, run_id):
        with create_session() as session:
            dag_run = session.query(DagRun).filter(DagRun.run_id == run_id).first()
            if dag_run is None:
                return False
            else:
                return True

    @property
    def airflow_client(self):
        if self._airflow_client is None:
            self._airflow_client = EventSchedulerClient(server_uri=self.config.notification_service_uri(),
                                                        namespace=SCHEDULER_NAMESPACE)
        return self._airflow_client

    def submit_workflow(self, workflow: Workflow, project_desc: ProjectDesc, args: Dict = None) -> WorkflowInfo:
        workflow_name = workflow.workflow_name
        dag_id = self.airflow_dag_id(project_desc.project_name, workflow.workflow_name)
        code_text = self.dag_generator.generator(workflow, dag_id, args)
        workflow.workflow_name = workflow_name
        deploy_path = self.config.properties().get('airflow_deploy_path')
        if deploy_path is None:
            raise Exception("airflow_deploy_path config not set!")
        if not os.path.exists(deploy_path):
            os.makedirs(deploy_path)
        airflow_file_path = os.path.join(deploy_path,
                                         dag_id + '.py')
        if os.path.exists(airflow_file_path):
            os.remove(airflow_file_path)
        with NamedTemporaryFile(mode='w+t', prefix=dag_id, suffix='.py', dir='/tmp', delete=False) as f:
            f.write(code_text)
        os.rename(f.name, airflow_file_path)
        self.airflow_client.trigger_parse_dag(airflow_file_path)
        return WorkflowInfo(namespace=project_desc.project_name, workflow_name=workflow.workflow_name)

    def delete_workflow(self, project_name: Text, workflow_name: Text) -> Optional[WorkflowInfo]:
        dag_id = self.airflow_dag_id(project_name, workflow_name)
        if not self.dag_exist(dag_id):
            return None
        deploy_path = self.config.properties().get('airflow_deploy_path')
        if deploy_path is None:
            raise Exception("airflow_deploy_path config not set!")
        airflow_file_path = os.path.join(deploy_path, dag_id + '.py')
        if os.path.exists(airflow_file_path):
            os.remove(airflow_file_path)

        # stop all workflow executions
        self.kill_all_workflow_execution(project_name, workflow_name)

        # clean db meta
        with create_session() as session:
            dag = session.query(DagModel).filter(DagModel.dag_id == dag_id).first()
            session.query(DagTag).filter(DagTag.dag_id == dag_id).delete()
            session.query(DagModel).filter(DagModel.dag_id == dag_id).delete()
            session.query(DagCode).filter(DagCode.fileloc_hash == DagCode.dag_fileloc_hash(dag.fileloc)).delete()
            session.query(SerializedDagModel).filter(SerializedDagModel.dag_id == dag_id).delete()
            session.query(DagRun).filter(DagRun.dag_id == dag_id).delete()
            session.query(TaskState).filter(TaskState.dag_id == dag_id).delete()
            session.query(TaskInstance).filter(TaskInstance.dag_id == dag_id).delete()
            session.query(TaskExecution).filter(TaskExecution.dag_id == dag_id).delete()
        return WorkflowInfo(namespace=project_name, workflow_name=workflow_name)

    def pause_workflow_scheduling(self, project_name: Text, workflow_name: Text) -> WorkflowInfo:
        dag_id = self.airflow_dag_id(project_name, workflow_name)
        DagModel.get_dagmodel(dag_id=dag_id).set_is_paused(is_paused=True)
        return WorkflowInfo(namespace=project_name, workflow_name=workflow_name)

    def resume_workflow_scheduling(self, project_name: Text, workflow_name: Text) -> WorkflowInfo:
        dag_id = self.airflow_dag_id(project_name, workflow_name)
        DagModel.get_dagmodel(dag_id=dag_id).set_is_paused(is_paused=False)
        return WorkflowInfo(namespace=project_name, workflow_name=workflow_name)

    def get_workflow(self, project_name: Text, workflow_name: Text) -> Optional[WorkflowInfo]:
        dag_id = self.airflow_dag_id(project_name, workflow_name)
        with create_session() as session:
            dag = session.query(DagModel).filter(DagModel.dag_id == dag_id).first()
            if dag is None:
                return None
            else:
                return WorkflowInfo(namespace=project_name, workflow_name=workflow_name)

    def list_workflows(self, project_name: Text) -> List[WorkflowInfo]:
        with create_session() as session:
            dag_list = session.query(DagModel).filter(DagModel.dag_id.startswith('{}.'.format(project_name))).all()
            if dag_list is None:
                return []
            else:
                result = []
                for dag in dag_list:
                    ns, workflow_name = self.parse_namespace_workflow_name(dag.dag_id)
                    result.append(WorkflowInfo(namespace=project_name, workflow_name=workflow_name))
                return result

    def start_new_workflow_execution(self, project_name: Text, workflow_name: Text) -> Optional[WorkflowExecutionInfo]:
        dag_id = self.airflow_dag_id(project_name, workflow_name)
        deploy_path = self.config.properties().get('airflow_deploy_path')
        if deploy_path is None:
            raise Exception("airflow_deploy_path config not set!")
        if not self.dag_exist(dag_id):
            return None
        context: ExecutionContext = self.airflow_client.schedule_dag(dag_id)
        return WorkflowExecutionInfo(workflow_info=WorkflowInfo(namespace=project_name, workflow_name=workflow_name),
                                     execution_id=context.dagrun_id,
                                     state=job_meta.State.INIT)

    def kill_all_workflow_execution(self, project_name: Text, workflow_name: Text) -> List[WorkflowExecutionInfo]:
        workflow_execution_list = self.list_workflow_executions(project_name, workflow_name)
        for we in workflow_execution_list:
            if we.state == job_meta.State.RUNNING:
                self.kill_workflow_execution(we.execution_id)
        return workflow_execution_list

    def kill_workflow_execution(self, execution_id: Text) -> Optional[WorkflowExecutionInfo]:
        with create_session() as session:
            dag_run = session.query(DagRun).filter(DagRun.run_id == execution_id).first()
            if dag_run is None:
                return None
            project_name, workflow_name = self.dag_id_to_namespace_workflow(dag_run.dag_id)
            context: ExecutionContext = ExecutionContext(execution_id)
            current_context = self.airflow_client.stop_dag_run(dag_run.dag_id, context)
            return WorkflowExecutionInfo(workflow_info=WorkflowInfo(namespace=project_name,
                                                                    workflow_name=workflow_name),
                                         execution_id=current_context.dagrun_id,
                                         state=job_meta.State.KILLING)

    def get_workflow_execution(self, execution_id: Text) -> Optional[WorkflowExecutionInfo]:
        with create_session() as session:
            dag_run = session.query(DagRun).filter(DagRun.run_id == execution_id).first()
            if dag_run is None:
                return None
            else:
                state = self.airflow_state_to_state(dag_run.state)
                project_name, workflow_name = self.dag_id_to_namespace_workflow(dag_run.dag_id)
                return WorkflowExecutionInfo(workflow_info=WorkflowInfo(namespace=project_name,
                                                                        workflow_name=workflow_name),
                                             execution_id=dag_run.run_id, state=state)

    def list_workflow_executions(self, project_name: Text, workflow_name: Text) -> List[WorkflowExecutionInfo]:
        dag_id = self.airflow_dag_id(project_name, workflow_name)
        with create_session() as session:
            dagrun_list = session.query(DagRun).filter(DagRun.dag_id == dag_id).all()
            if dagrun_list is None:
                return []
            else:
                result = []
                for dagrun in dagrun_list:
                    state = self.airflow_state_to_state(dagrun.state)
                    result.append(WorkflowExecutionInfo(workflow_info=WorkflowInfo(namespace=project_name,
                                                                                   workflow_name=workflow_name),
                                                        execution_id=dagrun.run_id, state=state))
                return result

    def start_job(self, job_name: Text, execution_id: Text) -> Optional[JobInfo]:
        with create_session() as session:
            dag_run = session.query(DagRun).filter(DagRun.run_id == execution_id).first()
            if dag_run is None:
                return None
            if dag_run.state != State.RUNNING:
                raise Exception('execution: {} state: {} can not trigger job.'.format(execution_id, dag_run.state))
            task = dag_run.get_task_instance(job_name, session)
            if task is None:
                return None
            if task.state in State.unfinished:
                raise Exception('job:{} state: {} can not start!'.format(job_name, task.state))
            self.airflow_client.schedule_task(dag_id=dag_run.dag_id,
                                              task_id=job_name,
                                              action=SchedulingAction.START,
                                              context=ExecutionContext(dagrun_id=dag_run.run_id))
            project_name, workflow_name = self.dag_id_to_namespace_workflow(dag_run.dag_id)
            return JobInfo(job_name=job_name,
                           state=self.airflow_state_to_state(task.state),
                           workflow_execution
                           =WorkflowExecutionInfo(workflow_info=WorkflowInfo(namespace=project_name,
                                                                             workflow_name=workflow_name),
                                                  execution_id=dag_run.run_id,
                                                  state=self.airflow_state_to_state(dag_run.state)))

    def stop_job(self, job_name: Text, execution_id: Text) -> Optional[JobInfo]:
        with create_session() as session:
            dag_run = session.query(DagRun).filter(DagRun.run_id == execution_id).first()
            if dag_run is None:
                return None
            if dag_run.state != State.RUNNING:
                raise Exception('execution: {} state: {} can not trigger job.'.format(execution_id, dag_run.state))
            task = dag_run.get_task_instance(job_name, session)
            if task is None:
                return None
            if task.state != State.RUNNING:
                raise Exception('job:{} state: {} can not stop!'.format(job_name, task.state))
            self.airflow_client.schedule_task(dag_id=dag_run.dag_id,
                                              task_id=job_name,
                                              action=SchedulingAction.STOP,
                                              context=ExecutionContext(dagrun_id=dag_run.run_id))
            project_name, workflow_name = self.dag_id_to_namespace_workflow(dag_run.dag_id)
            return JobInfo(job_name=job_name,
                           state=self.airflow_state_to_state(task.state),
                           workflow_execution
                           =WorkflowExecutionInfo(workflow_info=WorkflowInfo(namespace=project_name,
                                                                             workflow_name=workflow_name),
                                                  execution_id=dag_run.run_id,
                                                  state=self.airflow_state_to_state(dag_run.state)))

    def restart_job(self, job_name: Text, execution_id: Text) -> Optional[JobInfo]:
        with create_session() as session:
            dag_run = session.query(DagRun).filter(DagRun.run_id == execution_id).first()
            if dag_run is None:
                return None
            if dag_run.state != State.RUNNING:
                raise Exception('execution: {} state: {} can not trigger job.'.format(execution_id, dag_run.state))
            task = dag_run.get_task_instance(job_name, session)
            if task is None:
                return None
            self.airflow_client.schedule_task(dag_id=dag_run.dag_id,
                                              task_id=job_name,
                                              action=SchedulingAction.RESTART,
                                              context=ExecutionContext(dagrun_id=dag_run.run_id))
            project_name, workflow_name = self.dag_id_to_namespace_workflow(dag_run.dag_id)
            return JobInfo(job_name=job_name,
                           state=self.airflow_state_to_state(task.state),
                           workflow_execution
                           =WorkflowExecutionInfo(workflow_info=WorkflowInfo(namespace=project_name,
                                                                             workflow_name=workflow_name),
                                                  execution_id=dag_run.run_id,
                                                  state=self.airflow_state_to_state(dag_run.state)))

    def get_job(self, job_name: Text, execution_id: Text) -> Optional[JobInfo]:
        with create_session() as session:
            dag_run = session.query(DagRun).filter(DagRun.run_id == execution_id).first()
            if dag_run is None:
                return None
            task = session.query(TaskInstance).filter(TaskInstance.dag_id == dag_run.dag_id,
                                                      TaskInstance.execution_date == dag_run.execution_date,
                                                      TaskInstance.task_id == job_name).first()
            if task is None:
                return None
            else:
                project_name, workflow_name = self.dag_id_to_namespace_workflow(dag_run.dag_id)
                return JobInfo(job_name=job_name,
                               state=self.airflow_state_to_state(task.state),
                               workflow_execution
                               =WorkflowExecutionInfo(workflow_info=WorkflowInfo(namespace=project_name,
                                                                                 workflow_name=workflow_name),
                                                      execution_id=dag_run.run_id,
                                                      state=self.airflow_state_to_state(dag_run.state)))

    def list_jobs(self, execution_id: Text) -> List[JobInfo]:
        with create_session() as session:
            dag_run = session.query(DagRun).filter(DagRun.run_id == execution_id).first()
            if dag_run is None:
                return None
            task_list = session.query(TaskInstance).filter(TaskInstance.dag_id == dag_run.dag_id,
                                                           TaskInstance.execution_date == dag_run.execution_date).all()
            if task_list is None:
                return []
            else:
                result = []
                project_name, workflow_name = self.dag_id_to_namespace_workflow(dag_run.dag_id)
                for task in task_list:
                    job = JobInfo(job_name=task.task_id,
                                  state=self.airflow_state_to_state(task.state),
                                  workflow_execution
                                  =WorkflowExecutionInfo(workflow_info=WorkflowInfo(namespace=project_name,
                                                                                    workflow_name=workflow_name),
                                                         execution_id=dag_run.run_id,
                                                         state=self.airflow_state_to_state(dag_run.state)))
                    result.append(job)
                return result
