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
import os
from ai_flow.application_master.master_config import MasterConfig, DBType


class TestConfiguration(unittest.TestCase):

    def test_dump_load_configuration(self):
        config = MasterConfig()
        config.set_db_uri(db_type=DBType.SQLITE, uri="sqlite:///sql.db")
        self.assertEqual('sql.db', config.get_sql_lite_db_file())

    def test_load_master_configuration(self):
        config = MasterConfig()
        config.load_from_file(os.path.dirname(__file__) + '/master_config.yaml')
        self.assertEqual('sql_lite', config.get_db_type())
        self.assertEqual('/tmp/repo', config.get_scheduler_config()['repository'])
        self.assertEqual(False, config.start_scheduling_service())


if __name__ == '__main__':
    unittest.main()
