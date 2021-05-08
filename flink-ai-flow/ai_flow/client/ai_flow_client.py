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
from ai_flow.api.configuration import _default_project_config, ensure_project_registered
from ai_flow.api.execution import AirflowOperation
from ai_flow.rest_endpoint.service.client.aiflow_client import AIFlowClient

_default_ai_flow_client = None
_default_master_uri = 'localhost:50051'

_default_airflow_operation_client = None


def get_ai_flow_client():
    """ Get AI flow Client. """

    global _default_ai_flow_client, _default_master_uri
    ensure_project_registered()
    if _default_ai_flow_client is None:
        current_uri = _default_project_config.get_master_uri()
        if current_uri is None:
            return None
        else:
            _default_master_uri = current_uri
            _default_ai_flow_client \
                = AIFlowClient(server_uri=_default_master_uri,
                               notification_service_uri=_default_project_config.get_notification_service_uri())
            return _default_ai_flow_client
    else:
        current_uri = _default_project_config.get_master_uri()
        if current_uri != _default_master_uri:
            _default_master_uri = current_uri
            _default_ai_flow_client \
                = AIFlowClient(server_uri=_default_master_uri,
                               notification_service_uri=_default_project_config.get_notification_service_uri())

        return _default_ai_flow_client


def get_airflow_operation_client():
    """ Get a client to operate airflow dags and tasks. """
    global _default_airflow_operation_client
    ensure_project_registered()
    if _default_airflow_operation_client:
        return _default_airflow_operation_client
    else:
        return AirflowOperation(_default_project_config.get_notification_service_uri())
