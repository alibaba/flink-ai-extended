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
import uuid
import sys
import subprocess as sp
from typing import Text, Dict, Any, Optional, Tuple, List
import os
from pathlib import Path
import logging

from ai_flow.airflow.dag_generator import job_name_to_task_id
from ai_flow.common.json_utils import Jsonable

from ai_flow.workflow.job_config import PeriodicConfig

from ai_flow.graph.edge import DataEdge
from ai_flow.plugins.local_cmd_job_plugin import LocalJobHandler
from ai_flow.graph.ai_node import AINode
from ai_flow.graph.ai_nodes import Example
from ai_flow.log import log_path_utils
from ai_flow.plugins.job_plugin import AISubGraph, ProjectDesc, JobContext, \
    AbstractJobConfig, AbstractJob, AbstractEngine, AbstractPlatform
from ai_flow.plugins.local_cmd_job_plugin import LocalJobPlugin
from ai_flow.plugins.local_platform import LocalPlatform
from ai_flow.meta.job_meta import ExecutionMode
from ai_flow.graph.ai_nodes.executable import ExecutableNode
from ai_flow.project.blob_manager import BlobManagerFactory
from ai_flow.common.serialization_utils import serialize
from python_ai_flow.user_define_funcs import Executor
from python_ai_flow.python_engine import PythonEngine
from python_ai_flow.python_job_common import RunGraph, batch_run_func, stream_run_func, RunArgs, BaseComponent, \
    ExampleComponentRegistry, BaseExampleComponent, ExecuteComponent

ExampleType = "example_type"


class LocalPythonJobConfig(AbstractJobConfig):
    def __init__(self, job_name: Text = None,
                 periodic_config: PeriodicConfig = None, exec_mode: Optional[ExecutionMode] = ExecutionMode.BATCH,
                 properties: Dict[Text, Jsonable] = None) -> None:
        super().__init__(platform=LocalPlatform.platform(), engine=PythonEngine.engine(), job_name=job_name,
                         periodic_config=periodic_config,
                         exec_mode=exec_mode, properties=properties)


class LocalPythonJob(AbstractJob):
    def __init__(self,
                 run_func: bytes,
                 run_graph: Optional[RunGraph],
                 job_context: JobContext = JobContext(),
                 job_config: LocalPythonJobConfig = LocalPythonJobConfig()):
        super().__init__(job_context=job_context, job_config=job_config)
        self.run_func = run_func
        self.run_graph = run_graph
        self.exec_func_file: Text = None
        self.exec_args_file: Text = None


class LocalPythonJobPlugin(LocalJobPlugin):

    def __init__(self) -> None:
        super().__init__()
        self.job_handler_map: Dict[Text, LocalJobHandler] = {}
        self.example_registry: ExampleComponentRegistry = ExampleComponentRegistry()

    def register_example(self, key: Text, component: type(BaseExampleComponent)):
        self.example_registry.register(key, component)

    def compile_node(self, node: AINode, context: JobContext) -> Executor:
        exec_mode: ExecutionMode = context.get_execution_mode()

        def component_execute(component: BaseComponent):
            if exec_mode == ExecutionMode.BATCH:
                executor = component.batch_executor()
            else:
                executor = component.stream_executor()
            return executor

        executor: Executor = None
        if isinstance(node, Example):
            n: Example = node
            if node.executor is not None:
                example_type = 'udf'
            else:
                example_type = n.example_meta.data_type
            if example_type != 'udf' and example_type != "pandas" and example_type != "numpy":
                raise Exception("Base example node only supports pandas and numpy now")

            example_component_type: type(BaseExampleComponent) = self.example_registry.get_object(example_type)
            example_component = example_component_type(node, context)
            executor = component_execute(example_component)
        elif isinstance(node, ExecutableNode):
            n: ExecutableNode = node
            c = ExecuteComponent(n, context)
            executor = component_execute(c)
        else:
            pass
        return executor

    def build_run_graph(self, sub_graph: AISubGraph, context: JobContext) -> RunGraph:
        run_graph = RunGraph()
        processed_nodes = set()
        node_list: List[AINode] = []
        for n in sub_graph.nodes.values():
            node_list.append(n)
        for e in sub_graph.edges:
            data_channel_list = []
            for c in sub_graph.edges[e]:
                cc: DataEdge = c
                data_channel_list.append(cc)
            run_graph.dependencies[e] = data_channel_list

        node_size = len(sub_graph.nodes)
        processed_size = len(processed_nodes)
        while processed_size != node_size:
            p_nodes = []
            for i in range(len(node_list)):
                if node_list[i].instance_id in sub_graph.edges:
                    flag = True
                    for c in sub_graph.edges[node_list[i].instance_id]:
                        if c.target_node_id in processed_nodes:
                            pass
                        else:
                            flag = False
                            break
                else:
                    flag = True
                if flag:
                    p_nodes.append(node_list[i])
            if 0 == len(p_nodes):
                raise Exception("graph has circle!")
            for n in p_nodes:
                run_graph.nodes.append(n)
                run_graph.executor_bytes.append(serialize(self.compile_node(n, context=context)))
                node_list.remove(n)
                processed_nodes.add(n.instance_id)
            processed_size = len(processed_nodes)
        return run_graph

    def generate(self, sub_graph: AISubGraph, project_desc: ProjectDesc) -> LocalPythonJob:
        if sub_graph.config.exec_mode == ExecutionMode.BATCH:
            run_func = serialize(batch_run_func)
            py_context = JobContext(ExecutionMode.BATCH)
        else:
            run_func = serialize(stream_run_func)
            py_context = JobContext(ExecutionMode.STREAM)
        py_context.project_config = project_desc.project_config
        run_graph: RunGraph = self.build_run_graph(sub_graph, py_context)
        job_config: LocalPythonJobConfig = sub_graph.config
        return LocalPythonJob(run_graph=run_graph, run_func=run_func,
                              job_context=py_context, job_config=job_config)

    def generate_job_resource(self, job: LocalPythonJob):
        se_func = job.run_func
        se_args = serialize(RunArgs(run_graph=job.run_graph, job_context=job.job_context))
        project_path = job.job_config.project_path
        if project_path is None:
            project_path = "/tmp"
        project_path_temp = project_path + "/temp"
        if not os.path.exists(project_path_temp):
            os.mkdir(project_path_temp)
        func_file = project_path_temp + '/tmp_func' + str(uuid.uuid4()) + job.instance_id
        args_file = project_path_temp + '/tmp_args' + str(uuid.uuid4()) + job.instance_id
        with open(func_file, 'wb') as f:
            f.write(se_func)

        with open(args_file, 'wb') as f:
            f.write(se_args)
        job.exec_func_file: Text = Path(func_file).name
        job.exec_args_file: Text = Path(args_file).name

    def submit_job(self, job: LocalPythonJob) -> Any:
        """
        :param job:
        :return:
        """
        blob_manager = BlobManagerFactory.get_blob_manager(job.job_config.properties)
        copy_path = sys.path.copy()
        if job.job_config.project_path is not None:
            downloaded_blob_path = blob_manager.download_blob(job.instance_id, job.job_config.project_path)
            python_codes_path = downloaded_blob_path + '/python_codes'
            copy_path.append(python_codes_path)
        if job.job_config.project_desc.python_paths is not None:
            copy_path.extend(job.job_config.project_desc.python_paths)
        env = os.environ.copy()
        env['PYTHONPATH'] = ':'.join(copy_path)

        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        script_path = father_path + '/local_job_run.py'

        entry_module_path = job.job_config.properties['entry_module_path']
        python3_location = sys.executable
        cmd = [python3_location, script_path, job.job_config.project_path,
               job.exec_func_file, job.exec_args_file, entry_module_path]
        logging.info(' '.join(cmd))
        # every job submitter need set the job log file,
        # local python job set log file name LocalPythonJob_{workflow_execution_id}_{stdout,stderr}.log
        stdout_log = log_path_utils.stdout_log_path(job.job_config.project_desc.get_absolute_log_path(),
                                                    job.job_name)
        stderr_log = log_path_utils.stderr_log_path(job.job_config.project_desc.get_absolute_log_path(),
                                                    job.job_name)
        if not os.path.exists(job.job_config.project_desc.get_absolute_log_path()):
            os.mkdir(job.job_config.project_desc.get_absolute_log_path())

        with open(stdout_log, 'a') as out, open(stderr_log, 'a') as err:
            process = sp.Popen(cmd,
                               stderr=err, stdout=out,
                               shell=False, env=env)
            job_handler = LocalJobHandler(job_instance_id=job.instance_id,
                                          job_uuid=job.uuid,
                                          workflow_id=job.job_context.workflow_execution_id,
                                          process_object=process)
            self.job_handler_map[job.uuid] = job_handler
            return job_handler

    def is_finished(self, job: LocalPythonJob) -> Tuple[bool, Optional[int]]:
        """
        :param job:
        :return:
        """
        job_handler = self.job_handler_map[job.uuid]
        if job_handler.process_object is not None and job_handler.process_object.poll() is not None:
            return True, job_handler.process_object.returncode
        return False, None

    def cleanup_job(self, job: LocalPythonJob):
        """
        :param job: clean up the local python job resource when the workflow finished or failed
        :return:
        """
        pass

    def job_type(self) -> type(AbstractJob):
        return LocalPythonJob

    def job_config_type(self) -> type(AbstractJobConfig):
        return LocalPythonJobConfig

    def platform(self) -> type(AbstractPlatform):
        return LocalPlatform

    def engine(self) -> type(AbstractEngine):
        return PythonEngine

    def generate_code(self, op_index, job: AbstractJob):
        LOCAL_PYTHON_OPERATOR = """env_{0}={3}
op_{0} = BashOperator(task_id='{1}', dag=dag, bash_command='{2}', env=env_{0})\n"""
        copy_path = sys.path.copy()
        if job.job_config.project_desc.python_dependencies is not None:
            python_codes_path = job.job_config.project_desc.project_path + '/python_codes'
            copy_path.append(python_codes_path)
        if job.job_config.project_desc.python_paths is not None:
            copy_path.extend(job.job_config.project_desc.python_paths)
        current_path = os.path.abspath(__file__)
        python_package_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        script_path = python_package_path + '/local_job_run.py'

        entry_module_path = job.job_config.properties['entry_module_path']
        python3_location = sys.executable
        cmd = [python3_location, script_path, job.job_config.project_path,
               job.exec_func_file, job.exec_args_file, entry_module_path]
        cmd_str = ' '.join(cmd)
        add_path = ':'.join(copy_path)
        envs = os.environ.copy()
        envs['PYTHONPATH'] = add_path
        code_text = LOCAL_PYTHON_OPERATOR.format(op_index, job_name_to_task_id(job.job_name), cmd_str, envs)
        return code_text


_default_local_python_job_plugin = LocalPythonJobPlugin()


def get_default_local_python_job_plugin() -> LocalPythonJobPlugin:
    return _default_local_python_job_plugin


def register_local_example_component(key: Text, component: type(BaseExampleComponent)) -> None:
    _default_local_python_job_plugin.register_example(key, component)
