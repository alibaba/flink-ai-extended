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
import json
import logging
import os
import subprocess
import sys
import uuid
from typing import Dict, Any

import requests
from airflow.operators.bash_operator import BashOperator
from ai_flow.airflow.dag_generator import job_name_to_task_id
from ai_flow.common import serialization_utils
from ai_flow.common.json_utils import dumps
from ai_flow.log import log_path_utils
from ai_flow.meta.job_meta import ExecutionMode
from ai_flow.plugins.job_plugin import AISubGraph, ProjectDesc, JobContext, \
    AbstractJobConfig, AbstractJob, AbstractEngine
from ai_flow.plugins.language import LanguageType
from ai_flow.plugins.local_cmd_job_plugin import LocalCMDJobPlugin, LocalCMDJob
from ai_flow.plugins.local_platform import LocalPlatform, LocalJobHandler
from ai_flow.project.blob_manager import BlobManagerFactory
from ai_flow.project.project_util import file_path_to_absolute_module
from flink_ai_flow import version
from flink_ai_flow.flink_engine import FlinkEngine
from flink_ai_flow.flink_executor import FlinkPythonExecutor, FlinkJavaExecutor
from flink_ai_flow.flink_job_common import FlinkJobConfig


class LocalFlinkJobConfig(FlinkJobConfig):
    @staticmethod
    def from_dict(data: Dict, config) -> object:
        FlinkJobConfig.from_dict(data, config)
        config.local_mode = data.get('local_mode')
        return config

    def __init__(self):
        super().__init__(platform=LocalPlatform.platform())
        self.local_mode = 'python'


class LocalFlinkJob(LocalCMDJob):
    """
    Implementation of LocalCMDJob. Represents the local flink job.
    """

    def __init__(self,
                 ai_graph,
                 job_context: JobContext,
                 job_config: LocalFlinkJobConfig):
        """
        Construct method of LocalFlinkJob.

        :param ai_graph: Graph generated from ai nodes.
        :param job_context: Context of local flink job.
        :param job_config: Config of local flink job.
        """
        super().__init__(exec_cmd=None, job_context=job_context, job_config=job_config)
        self.ai_graph = ai_graph
        self.config_file = None


class LocalFlinkJobPlugin(LocalCMDJobPlugin):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_language_type(sub_graph: AISubGraph) -> LanguageType:
        """
        Get language type, if JAVA, the job is java flink job,
        if PYTHON, the job is python flink job.

        :param sub_graph: Graph generates from ai nodes.
        :return: JAVA or PYTHON.
        """
        for node in sub_graph.nodes.values():
            try:
                if isinstance(node.executor, FlinkPythonExecutor):
                    return LanguageType.PYTHON
                elif isinstance(node.executor, FlinkJavaExecutor):
                    return LanguageType.JAVA
            except Exception:
                pass
        return LanguageType.PYTHON

    def generate(self, sub_graph: AISubGraph, project_desc: ProjectDesc) -> AbstractJob:
        """
        Generate local flink job.

        :param sub_graph: Sub graph generates from ai nodes.
        :param project_desc: Description of the project.
        :return: Base job Object.
        """
        if sub_graph.config.exec_mode == ExecutionMode.BATCH:
            flink_context = JobContext(ExecutionMode.BATCH)
        else:
            flink_context = JobContext(ExecutionMode.STREAM)

        flink_context.project_config = project_desc.project_config
        local_flink_job_config: LocalFlinkJobConfig = sub_graph.config
        job = LocalFlinkJob(ai_graph=sub_graph, job_context=flink_context, job_config=local_flink_job_config)

        if job.job_config.language_type is None:
            job.job_config.language_type = self.get_language_type(sub_graph)
        # set jar and class
        job.job_config.main_class = version.main_class
        job.job_config.jar_path = version.jar_path

        # set python main
        job.job_config.py_entry_file = version.py_main_file
        return job

    def generate_job_resource(self, job: LocalFlinkJob) -> None:
        """
        Generate flink job resource.

        :param job: Local flink job.
        """
        # gen config file
        project_path = job.job_config.project_path
        if project_path is None:
            project_path = "/tmp"
        project_path_temp = project_path + "/temp"
        if not os.path.exists(project_path_temp):
            os.mkdir(project_path_temp)

        if job.job_config.language_type == LanguageType.JAVA:
            execution_config_file = project_path_temp + '/job_execution_config_' + str(
                uuid.uuid4()) + "_" + job.instance_id

            with open(execution_config_file, 'w') as f:
                f.write(dumps(job))
            job.config_file = execution_config_file

        else:
            execution_config_file = '/job_execution_config_' + str(
                uuid.uuid4()) + "_" + job.instance_id
            real_execution_config_file = project_path_temp + execution_config_file
            with open(real_execution_config_file, 'wb') as f:
                f.write(serialization_utils.serialize(job))
            job.config_file = execution_config_file

    def is_finished(self, job) -> Any:
        """
        If the job has finished.

        :param job: A flink job object which contains the necessary information for an execution.
        :return: True if the job has not finished yet, and the returncode of the job,
                 False if the job has finished, None represents the job has exited.
        """
        if job.exec_handle is not None and job.exec_handle.poll() is not None:
            return True, job.exec_handle.returncode
        return False, None

    def submit_job(self, job: LocalFlinkJob):
        """
        Submit the flink job to run in local.

        :param job: A flink job object which contains the necessary information for an execution.
        :return: A job handler that maintains the handler of a job in runtime.
        """
        # generate cmd
        if job.job_config.language_type == LanguageType.JAVA:
            exec_cmd = ['flink', 'run']
            exec_cmd.extend(['-m', job.job_config.jm_host_port])
            if job.job_config.class_path is not None:
                exec_cmd.extend(['-C', job.job_config.class_path])

            if job.job_config.project_desc.jar_dependencies is not None:
                for jar in job.job_config.project_desc.jar_dependencies:
                    exec_cmd.extend(['-C', "file://{}".format(jar)])
            if job.job_config.main_class is not None:
                exec_cmd.extend(['-c', job.job_config.main_class])

            exec_cmd.extend([job.job_config.jar_path])
            exec_cmd.extend(['--execution-config', job.config_file])

            if job.job_config.args is not None:
                exec_cmd.extend(job.job_config.args)
        else:
            if 'entry_module_path' not in job.job_config.project_desc.project_config:
                entry_module_path = (file_path_to_absolute_module(sys.argv[0])).split('.')[-1]
            else:
                entry_module_path = job.job_config.project_desc.project_config['entry_module_path']

            python3_location = sys.executable
            if job.job_config.local_mode == 'python':
                exec_cmd = [python3_location, version.py_main_file, job.job_config.project_path,
                            job.config_file, entry_module_path]
            else:
                exec_cmd = ['flink', 'run',
                            '-pym', version.py_cluster_module,
                            '-pyfs', job.job_config.project_path + ',' + job.job_config.project_path + '/python_codes/',
                            '-pyexec', python3_location,
                            '--project-path', job.job_config.project_path,
                            '--config-file', job.config_file,
                            '--entry-module-path', entry_module_path]

        job.exec_cmd = exec_cmd
        logging.info(' '.join(exec_cmd))

        sys_env = os.environ.copy()
        if job.job_config.flink_home is not None:
            sys_env['PATH'] = job.job_config.flink_home + '/bin:' + sys_env['PATH']
        blob_manager = BlobManagerFactory.get_blob_manager(job.job_config.properties)
        copy_path = sys.path.copy()
        if job.job_config.project_path is not None:
            downloaded_blob_path = blob_manager.download_blob(job.instance_id, job.job_config.project_path)
            python_codes_path = downloaded_blob_path + '/python_codes'
            copy_path.append(python_codes_path)
        if job.job_config.project_desc.python_paths is not None:
            copy_path.extend(job.job_config.project_desc.python_paths)
        sys_env['PYTHONPATH'] = ':'.join(copy_path)
        logging.info(sys_env['PYTHONPATH'])
        # every job submitter need set the job log file,
        # local flink job set log file name LocalFlinkJob_{workflow_execution_id}_{stdout,stderr}.log

        stdout_log = log_path_utils.stdout_log_path(job.job_config.project_desc.get_absolute_log_path(),
                                                    job.job_name)
        stderr_log = log_path_utils.stderr_log_path(job.job_config.project_desc.get_absolute_log_path(),
                                                    job.job_name)
        if not os.path.exists(job.job_config.project_desc.get_absolute_log_path()):
            os.mkdir(job.job_config.project_desc.get_absolute_log_path())
        sys_env['PYFLINK_CLIENT_EXECUTABLE'] = sys.executable
        with open(stdout_log, 'a') as out, open(stderr_log, 'a') as err:
            submitted_process = subprocess.Popen(
                args=job.exec_cmd,
                shell=False,
                stdout=out,
                stderr=err,
                env=sys_env
            )
        exec_handle = LocalJobHandler(job_uuid=job.uuid,
                                      job_instance_id=job.instance_id,
                                      workflow_id=job.job_context.workflow_execution_id,
                                      process_object=submitted_process)
        self.job_handler_map[job.uuid] = exec_handle
        return exec_handle

    def job_type(self) -> type(AbstractJob):
        return LocalFlinkJob

    def job_config_type(self) -> type(AbstractJobConfig):
        return LocalFlinkJobConfig

    def engine(self) -> type(AbstractEngine):
        return FlinkEngine

    def generate_code(self, op_index, job: LocalFlinkJob):
        # set env
        sys_env = os.environ.copy()
        if job.job_config.flink_home is not None:
            sys_env['PATH'] = job.job_config.flink_home + '/bin:' + sys_env['PATH']
        python_path = sys.path.copy()
        if job.job_config.project_path is not None:
            python_path.append(job.job_config.project_path + '/python_codes')
        if job.job_config.project_desc.python_paths is not None:
            python_path.extend(job.job_config.project_desc.python_paths)
        sys_env['PYTHONPATH'] = ':'.join(python_path)
        sys_env['PYFLINK_CLIENT_EXECUTABLE'] = sys.executable

        # generate cmd
        if job.job_config.language_type == LanguageType.JAVA:
            exec_cmd = ['flink', 'run']
            exec_cmd.extend(['-m', job.job_config.jm_host_port])
            if job.job_config.class_path is not None:
                exec_cmd.extend(['-C', job.job_config.class_path])

            if job.job_config.project_desc.jar_dependencies is not None:
                for jar in job.job_config.project_desc.jar_dependencies:
                    exec_cmd.extend(['-C', "file://{}".format(jar)])
            if job.job_config.main_class is not None:
                exec_cmd.extend(['-c', job.job_config.main_class])

            exec_cmd.extend([job.job_config.jar_path])
            exec_cmd.extend(['--execution-config', job.config_file])

            if job.job_config.args is not None:
                exec_cmd.extend(job.job_config.args)
        else:
            entry_module_path = job.job_config.properties['entry_module_path']
            python3_location = sys.executable
            if job.job_config.local_mode == 'python':
                exec_cmd = [python3_location, version.py_main_file, job.job_config.project_path,
                            job.config_file, entry_module_path]
            else:
                exec_cmd = ['flink', 'run',
                            '-pym', version.py_cluster_module,
                            '-pyfs', job.job_config.project_path + ',' + job.job_config.project_path + '/python_codes/',
                            '-pyexec', python3_location,
                            '--project-path', job.job_config.project_path,
                            '--config-file', job.config_file,
                            '--entry-module-path', entry_module_path]

        job.exec_cmd = exec_cmd
        logging.info(' '.join(exec_cmd))
        # generate code
        if job.job_config.language_type == LanguageType.PYTHON and job.job_config.local_mode == 'python':
            return """env_{0}={3}
op_{0} = BashOperator(task_id='{1}', bash_command='{2}', dag=dag, env=env_{0})\n""".format(
                op_index,
                job_name_to_task_id(job.job_name),
                ' '.join(job.exec_cmd),
                sys_env)
        else:
            return """from flink_ai_flow.local_flink_job import LocalFlinkOperator
env_{0}={4}
op_{0} = LocalFlinkOperator(task_id='{1}', bash_command='{2}', properties='{3}', dag=dag, env=env_{0})\n""".format(
                op_index,
                job_name_to_task_id(job.job_name),
                ' '.join(job.exec_cmd),
                json.dumps({'project_path': job.job_config.project_path,
                            'workflow_execution_id': job.job_context.workflow_execution_id,
                            'instance_id': job.instance_id}),
                sys_env)

    def stop_job(self, job: AbstractJob):
        if job.uuid is None or job.uuid not in self.job_handler_map:
            return
        logging.info(
            'stop {} {} {} {}'.format(job.job_context.workflow_execution_id, job.uuid, job.instance_id, job.job_name))
        job_execution_path = '{}/temp/{}/{}'.format(job.job_config.project_path, job.job_context.workflow_execution_id,
                                                    job.instance_id)
        if os.path.exists(job_execution_path):
            with open(job_execution_path) as f:
                job_id = f.readline()
            rest_url = os.environ.get('REST_URL') if 'REST_URL' in os.environ else 'http://localhost:8081'
            response = requests.patch('%s/jobs/%s' % (rest_url, job_id))
            if response.status_code == 200:
                os.remove(job_execution_path)


class LocalFlinkOperator(BashOperator):

    def __init__(
            self,
            properties,
            *args, **kwargs):
        super(LocalFlinkOperator, self).__init__(*args, **kwargs)
        self.properties = json.loads(properties)

    def on_kill(self):
        job_execution_path = '{}/temp/{}/{}'.format(self.properties['project_path'],
                                                    self.properties['workflow_execution_id'],
                                                    self.properties['instance_id'])
        logging.info('Job execution path: {}'.format(job_execution_path))
        if os.path.exists(job_execution_path):
            with open(job_execution_path) as f:
                job_id = f.readline()
            rest_url = os.environ.get('REST_URL') if 'REST_URL' in os.environ else 'http://localhost:8081'
            response = requests.patch('%s/jobs/%s' % (rest_url, job_id))
            if response.status_code == 200:
                os.remove(job_execution_path)
