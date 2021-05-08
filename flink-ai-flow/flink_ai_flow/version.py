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
main_class = 'com.apache.flink.ai.flow.FlinkJobMain'
jar_name = 'flink_ai_flow-1.0-SNAPSHOT.jar'
jar_path = os.path.dirname(os.path.abspath(__file__)) + '/flink_ai_flow-1.0-SNAPSHOT.jar'

py_main_file = os.path.dirname(os.path.abspath(__file__)) + '/pyflink/pyflink_job_main.py'
py_cluster_main_file = os.path.dirname(os.path.abspath(__file__)) + '/pyflink/pyflink_cluster_main.py'
py_cluster_module = 'flink_ai_flow.pyflink.pyflink_cluster_main'
