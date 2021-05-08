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
import unittest

from ai_flow.workflow.job_config import BaseJobConfig, PeriodicConfig

from ai_flow.airflow.dag_generator import DAGGenerator
from ai_flow.project.project_description import ProjectDesc
from ai_flow.meta.job_meta import State
from ai_flow.graph.edge import JobControlEdge, MetConfig, generate_job_status_key
from ai_flow.plugins.local_dummy_job_plugin import LocalDummyJob, SendEventJobConfig
from ai_flow.plugins.local_cmd_job_plugin import LocalCMDJob
from ai_flow.workflow.job import BaseJob
from notification_service.base_notification import UNDEFINED_EVENT_TYPE
from ai_flow.workflow.workflow import Workflow


class TestDAGGenerator(unittest.TestCase):

    @staticmethod
    def create_dummy_job(index) -> BaseJob:
        job: BaseJob = LocalDummyJob()
        job.job_context.workflow_execution_id = 1
        job.instance_id = str(index) + "_job"
        job.uuid = index
        job.job_name = job.instance_id
        return job

    @staticmethod
    def create_workflow() -> Workflow:
        workflow = Workflow()
        workflow.project_desc = ProjectDesc()
        workflow.project_desc.project_name = "workflow_1"
        for i in range(6):
            job = TestDAGGenerator.create_dummy_job(i)
            if i == 2:
                job.job_config = SendEventJobConfig('localhost:50051', 'key_1', 'value_1', UNDEFINED_EVENT_TYPE)
            elif i == 3:
                job.job_config = SendEventJobConfig('localhost:50051', 'key_2', 'value_2', UNDEFINED_EVENT_TYPE)
            elif i == 5:
                job.job_config = SendEventJobConfig('localhost:50051', 'key_2', 'value_2', "STOP_SCHEDULER_CMD")
            workflow.add_job(job)
        deps = []
        deps.append(JobControlEdge(target_node_id='0_job', source_node_id='2_job',
                                   met_config=MetConfig(event_key=generate_job_status_key('0_job'),
                                                        event_value=State.FINISHED.value)))
        deps.append(JobControlEdge(target_node_id='1_job', source_node_id='2_job',
                                   met_config=MetConfig(event_key=generate_job_status_key('1_job'),
                                                        event_value=State.FINISHED.value)))
        workflow.add_edges("2_job", deps)

        deps = []
        deps.append(JobControlEdge(target_node_id='2_job', source_node_id='4_job',
                                   met_config=MetConfig(event_key='key_1',
                                                        event_value='value_1',
                                                        event_type=UNDEFINED_EVENT_TYPE)))
        deps.append(JobControlEdge(target_node_id='3_job', source_node_id='4_job',
                                   met_config=MetConfig(event_key='key_2',
                                                        event_value='value_2',
                                                        event_type=UNDEFINED_EVENT_TYPE)))
        workflow.add_edges("4_job", deps)

        deps = []
        deps.append(JobControlEdge(target_node_id='4_job', source_node_id='5_job',
                                   met_config=MetConfig(event_key=generate_job_status_key('5_job'),
                                                        event_value=State.FINISHED.value)))
        workflow.add_edges("5_job", deps)
        workflow.workflow_id = 1
        return workflow

    def test_generate_dummy_dag_code(self):
        generator = DAGGenerator()
        workflow = TestDAGGenerator.create_workflow()
        dag = generator.generator(workflow)
        self.assertFalse('DummyOperator' in dag)
        self.assertFalse('SendEventOperator' in dag)

    @staticmethod
    def create_bash_job(index) -> BaseJob:
        job: BaseJob = LocalCMDJob(exec_cmd=['echo "{0} hello word!"'.format(index)])
        job.job_context.workflow_execution_id = 1
        job.instance_id = str(index) + "_job"
        job.name = str(index) + "_job"
        job.uuid = index
        job.job_name = job.instance_id
        job.job_config = BaseJobConfig()
        if 0 == index:
            job.job_config.periodic_config = PeriodicConfig(periodic_type='cron', args='* * * * *')
        elif 1 == index:
            job.job_config.periodic_config = PeriodicConfig(periodic_type='interval',
                                                            args={'seconds': 20, 'minutes': 1})
        return job

    @staticmethod
    def create_bash_workflow() -> Workflow:
        workflow = Workflow()
        workflow.project_desc = ProjectDesc()
        workflow.project_desc.project_name = "workflow_1"
        for i in range(3):
            job = TestDAGGenerator.create_bash_job(i)
            workflow.add_job(job)
        deps_1 = [JobControlEdge(target_node_id='', source_node_id='1_job',
                                 met_config=MetConfig(event_key='key_1',
                                                      event_value='value_1',
                                                      event_type=UNDEFINED_EVENT_TYPE))]
        deps_2 = [JobControlEdge(target_node_id='', source_node_id='2_job',
                                 met_config=MetConfig(event_key='key_2',
                                                      event_value='value_2',
                                                      event_type=UNDEFINED_EVENT_TYPE))]

        workflow.add_edges("1_job", deps_1)
        workflow.add_edges("2_job", deps_2)
        workflow.workflow_id = 1
        return workflow

    def test_generate_bash_dag_code(self):
        from datetime import datetime, timedelta
        default_args = {
            'owner': 'airflow',
            'depends_on_past': False,
            'start_date': datetime(2015, 12, 1),
            'email': ['airflow@example.com'],
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
            'schedule_interval': None,
        }
        generator = DAGGenerator()
        workflow = TestDAGGenerator.create_bash_workflow()
        dag = generator.generator(workflow, 'aa', default_args)
        print("\n\n")
        print(dag)
