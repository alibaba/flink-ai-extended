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
from typing import List

from ai_flow.common.path_util import get_file_dir, get_parent_dir

import ai_flow as af
from ai_flow import AIFlowMaster
from flink_ai_flow.pyflink import Executor, FlinkFunctionContext, SourceExecutor, SinkExecutor, TableEnvCreator
import test_util
from pyflink.table import DataTypes
from pyflink.table.descriptors import Schema, OldCsv, FileSystem
from pyflink.table import Table
import flink_ai_flow as faf
import os


class Transformer(Executor):
    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        return [input_list[0].group_by('word').select('word, count(1)')]


class Source(SourceExecutor):
    def execute(self, function_context: FlinkFunctionContext) -> Table:
        example_meta: af.ExampleMeta = function_context.get_example_meta()
        t_env = function_context.get_table_env()
        t_env.connect(FileSystem().path(example_meta.batch_uri)) \
            .with_format(OldCsv()
                         .field('word', DataTypes.STRING())) \
            .with_schema(Schema()
                         .field('word', DataTypes.STRING())) \
            .create_temporary_table('mySource')
        return t_env.from_path('mySource')


class Sink(SinkExecutor):
    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        example_meta: af.ExampleMeta = function_context.get_example_meta()
        output_file = example_meta.batch_uri
        if os.path.exists(output_file):
            os.remove(output_file)

        t_env = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        t_env.connect(FileSystem().path(output_file)) \
            .with_format(OldCsv()
                         .field_delimiter('\t')
                         .field('word', DataTypes.STRING())
                         .field('count', DataTypes.BIGINT())) \
            .with_schema(Schema()
                         .field('word', DataTypes.STRING())
                         .field('count', DataTypes.BIGINT())) \
            .create_temporary_table('mySink')
        statement_set.add_insert('mySink', input_table)


class TestPyFlinkJob():
    @classmethod
    def setUpClass(cls) -> None:
        config_file = test_util.get_master_config_file()
        cls.master = AIFlowMaster(config_file=config_file)
        cls.master.start()
        test_util.set_project_config(__file__)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.master.stop()
        af.unset_project_config()

    def tearDown(self):
        TestPyFlinkJob.master._clear_db()

    def test_run_pyflink_job(self):
        project_path = os.path.dirname(__file__) + '/../'
        af.set_project_config_file(project_path + "project.yaml")
        input_file = get_parent_dir(get_file_dir(__file__)) + '/resources/word_count.txt'
        output_file = get_file_dir(__file__) + "/word_count_output.csv"
        if os.path.exists(output_file):
            os.remove(output_file)

        example_1 = af.create_example(name="example_1",
                                      support_type=af.ExampleSupportType.EXAMPLE_BOTH,
                                      batch_uri=input_file,
                                      stream_uri=input_file,
                                      data_format="csv")

        example_2 = af.create_example(name="example_2",
                                      support_type=af.ExampleSupportType.EXAMPLE_BOTH,
                                      batch_uri=output_file,
                                      stream_uri=output_file,
                                      data_format="csv")
        flink_config = faf.LocalFlinkJobConfig()
        flink_config.local_mode = 'cluster'
        flink_config.flink_home = os.environ.get('FLINK_HOME')
        flink_config.set_table_env_create_func(TableEnvCreator())
        with af.config(flink_config):
            input_example = af.read_example(example_info=example_1,
                                            executor=faf.flink_executor.FlinkPythonExecutor(python_object=Source())
                                            )
            processed = af.transform(input_data_list=[input_example],
                                     executor=faf.flink_executor.FlinkPythonExecutor(python_object=Transformer()))

            af.write_example(input_data=processed,
                             example_info=example_2,
                             executor=faf.flink_executor.FlinkPythonExecutor(python_object=Sink())
                             )
        workflow_id = af.run(project_path)
        res = af.wait_workflow_execution_finished(workflow_id)

if __name__ == '__main__':

    tt = TestPyFlinkJob()
    tt.test_run_pyflink_job()
