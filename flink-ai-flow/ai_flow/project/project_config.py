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
from ai_flow.common.configuration import AIFlowConfiguration


class ProjectConfig(AIFlowConfiguration):
    def get_master_ip(self):
        return self["master_ip"]

    def set_master_ip(self, value):
        self["master_ip"] = value

    def get_master_port(self):
        return self["master_port"]

    def set_master_port(self, value):
        self["master_port"] = value

    def get_master_uri(self):
        return "{}:{}".format(self["master_ip"], self["master_port"])

    def get_project_name(self):
        return self["project_name"]

    def set_project_name(self, value):
        self["project_name"] = value

    def get_project_uuid(self):
        if "project_uuid" in self:
            return self["project_uuid"]
        else:
            return None

    def set_project_uuid(self, value):
        self["project_uuid"] = value

    def get_airflow_deploy_path(self):
        if "airflow_deploy_path" in self:
            return self["airflow_deploy_path"]
        else:
            return None

    def set_airflow_deploy_path(self, path):
        self["airflow_deploy_path"] = path

    def get_notification_service_uri(self):
        if "notification_uri" in self:
            return self["notification_uri"]
        else:
            return self.get_master_uri()

    def set_notification_service_uri(self, uri):
        self["notification_uri"] = uri

    def get_uploaded_project_path(self):
        if "uploaded_project_path" in self:
            return self["uploaded_project_path"]
        else:
            return self.get_master_uri()

    def set_uploaded_project_path(self, uri):
        self["uploaded_project_path"] = uri

    def get_schedule_interval(self):
        if "schedule_interval" in self:
            return self["schedule_interval"]
        else:
            return "@once"

    def set_schedule_interval(self, schedule_interval):
        self["schedule_interval"] = schedule_interval

    def get_enable_ha(self):
        if "enable_ha" in self:
            raw = self["enable_ha"]
            assert str(raw).strip().lower() in {"true", "false"}
            return bool(str(raw).strip().lower())
        else:
            return False

    def set_enable_ha(self, enable_ha):
        assert str(enable_ha).strip().lower() in {"true", "false"}
        self["enable_ha"] = enable_ha

    def _get_time_interval_ms(self, key, default):
        if key in self:
            raw = self[key]
            assert int(str(raw).strip()) >= 0
            return int(str(raw).strip())
        else:
            return default

    def _set_time_interval_ms(self, key, value):
        assert int(str(value).strip()) >= 0
        self[key] = value

    def get_list_member_interval_ms(self):
        return self._get_time_interval_ms("list_member_interval_ms", 5000)

    def set_list_member_interval_ms(self, list_member_interval_ms):
        self._set_time_interval_ms("list_member_interval_ms", list_member_interval_ms)

    def get_retry_interval_ms(self):
        return self._get_time_interval_ms("retry_interval_ms", 1000)

    def set_retry_interval_ms(self, retry_interval_ms):
        self._set_time_interval_ms("retry_interval_ms", retry_interval_ms)

    def get_retry_timeout_ms(self):
        return self._get_time_interval_ms("retry_timeout_ms", 10000)

    def set_retry_timeout_ms(self, retry_timeout_ms):
        self._set_time_interval_ms("retry_timeout_ms", retry_timeout_ms)


_default_project_config = ProjectConfig()
