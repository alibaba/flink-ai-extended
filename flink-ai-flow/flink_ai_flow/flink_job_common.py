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
import os
from ai_flow.common import serialization_utils
from ai_flow.plugins.job_plugin import AbstractJobConfig
from typing import Dict, Text
from ai_flow.plugins.language import LanguageType
from flink_ai_flow.pyflink.user_define_executor import TableEnvCreator
from flink_ai_flow.flink_engine import FlinkEngine


class FlinkJobConfig(AbstractJobConfig):
    @staticmethod
    def from_dict(data: Dict, config) -> object:
        AbstractJobConfig.from_dict(data, config)
        config.flink_home = data.get('flink_home', os.environ.get('FLINK_HOME'))
        config.jm_host_port = data.get('jm_host_port', 'localhost:8081')
        config.class_path = data.get('class_path')
        config.py_entry_file = data.get('py_entry_file')
        config.py_files = data.get('py_files')
        config.py_module = data.get('py_module')
        config.jar_path = data.get('jar_path')
        config.language_type: LanguageType = LanguageType(data.get('language_type', 'PYTHON'))
        config.flink_conf = data.get('flink_conf')
        config.image = data.get('ai_flow_worker_image')
        return config

    def __init__(self, platform: Text):
        super().__init__(platform=platform, engine=FlinkEngine.engine())
        self.flink_home = None
        self.jm_host_port = 'localhost:8081'
        self.class_path = None
        self.py_entry_file = None
        self.py_files = None
        self.py_module = None
        self.jar_path = None
        self.args = []
        self.language_type: LanguageType = LanguageType.PYTHON
        self.table_env_create_func: bytes = serialization_utils.serialize(TableEnvCreator())

    def set_table_env_create_func(self, func):
        self.table_env_create_func = serialization_utils.serialize(func)

    def get_table_env_create_func(self):
        return serialization_utils.deserialize(self.table_env_create_func)
