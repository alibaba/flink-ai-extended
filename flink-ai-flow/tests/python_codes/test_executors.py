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
from typing import List
import os
from ai_flow.udf.function_context import FunctionContext
from flink_ai_flow.pyflink import Executor, FlinkFunctionContext, SourceExecutor, SinkExecutor
from pyflink.table import DataTypes
from pyflink.table.descriptors import Schema, OldCsv, FileSystem
from pyflink.table import Table
import ai_flow as af


class HelloExecutor(Executor):
    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        print("hello world!")
        return []


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
