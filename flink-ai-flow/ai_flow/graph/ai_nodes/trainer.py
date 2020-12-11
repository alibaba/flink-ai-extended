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
from typing import Text, List
from ai_flow.meta.model_meta import ModelMeta
from ai_flow.common.properties import ExecuteProperties
from ai_flow.executor.executor import BaseExecutor
from ai_flow.graph.ai_nodes.executable import ExecutableNode
from ai_flow.graph.channel import Channel, NoneChannel


class Trainer(ExecutableNode):

    def __init__(self,
                 name: Text,
                 executor: BaseExecutor,
                 output_model: ModelMeta,
                 base_model: ModelMeta = None,
                 properties: ExecuteProperties = None,
                 instance_id=None,
                 output_num=0) -> None:
        super().__init__(
            executor=executor,
            properties=properties,
            name=name,
            instance_id=instance_id,
            output_num=output_num)
        self.output_model = output_model
        self.from_model_version = base_model
