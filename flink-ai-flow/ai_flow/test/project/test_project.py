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

from ai_flow.common.path_util import get_file_dir
from ai_flow.project.project_description import get_project_description_from
from ai_flow.api.configuration import set_project_config_file


class TestProjectConfig(unittest.TestCase):

    def test_load_project_config(self):
        project_path = get_file_dir(__file__)
        set_project_config_file(project_path+"/project.yaml")
        project_desc = get_project_description_from(project_path)
        self.assertEqual(project_desc.project_config.get_master_uri(), "localhost:50051")
        self.assertIsNone(project_desc.project_config.get('ai_flow config', None))
        self.assertEqual(project_desc.project_config['ai_flow_home'], '/opt/ai_flow')
        self.assertEqual(project_desc.project_config['ai_flow_job_master.host'], 'localhost')
        self.assertEqual(project_desc.project_config['ai_flow_job_master.port'], 8081)
        self.assertEqual(project_desc.project_config['ai_flow_conf'], 'taskmanager.slot=2')


if __name__ == '__main__':
    unittest.main()
