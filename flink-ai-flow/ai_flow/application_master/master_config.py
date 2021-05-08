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
from enum import Enum
from typing import Text


class DBType(str, Enum):
    SQLITE = "sql_lite"
    MYSQL = "mysql"
    MONGODB = "mongodb"

    @staticmethod
    def value_of(db_type):
        if db_type in ('SQL_LITE', 'sql_lite', 'sqlite'):
            return DBType.SQLITE
        elif db_type in ('MYSQL', 'mysql'):
            return DBType.MYSQL
        elif db_type in ('MONGODB', 'mongodb'):
            return DBType.MONGODB
        else:
            raise NotImplementedError


class MasterConfig(AIFlowConfiguration):
    def get_sql_lite_db_file(self):
        db_uri = self.get_db_uri()
        return db_uri[10:]

    def get_master_ip(self):
        return self["master_ip"]

    def get_master_port(self):
        return self["master_port"]

    def get_db_uri(self):
        return self["db_uri"]

    def get_db_type(self) -> DBType:
        return DBType.value_of(self["db_type"])

    def get_enable_ha(self):
        if "enable_ha" in self:
            raw = self.get("enable_ha")
            assert str(raw).strip().lower() in {"true", "false"}
            return bool(str(raw).strip().lower())
        else:
            return False

    def get_ha_ttl_ms(self):
        if "ha_ttl_ms" in self:
            return int(self.get("ha_ttl_ms"))
        else:
            return 10000

    def set_master_port(self, value):
        self["master_port"] = value

    def set_master_ip(self, value):
        self["master_ip"] = value

    def set_db_uri(self, db_type: DBType, uri: Text):
        self["db_type"] = db_type.value
        self["db_uri"] = uri

    def set_enable_ha(self, enable_ha):
        assert str(enable_ha).strip().lower() in {"true", "false"}
        self["enable_ha"] = str(enable_ha)

    def set_ha_ttl_ms(self, ha_ttl_ms):
        assert int(str(ha_ttl_ms).strip()) >= 0
        self["ha_ttl_ms"] = str(ha_ttl_ms).strip()

    def start_default_notification(self)->bool:
        if "start_default_notification" in self and self['start_default_notification'] is False:
            return False
        else:
            return True

    def set_notification_uri(self, notification_uri):
        self['notification_uri'] = notification_uri

    def get_notification_uri(self):
        return self.get('notification_uri')

    def get_scheduler_config(self):
        return self.get('scheduler')

    def set_scheduler_config(self, value):
        self['scheduler'] = value

    def start_meta_service(self):
        if "start_meta_service" in self and self['start_meta_service'] is False:
            return False
        else:
            return True

    def start_model_center_service(self):
        if "start_model_center_service" in self and self['start_model_center_service'] is False:
            return False
        else:
            return True

    def start_metric_service(self):
        if "start_metric_service" in self and self['start_metric_service'] is False:
            return False
        else:
            return True

    def start_deploy_service(self):
        if "start_deploy_service" in self and self['start_deploy_service'] is False:
            return False
        else:
            return True

    def start_scheduling_service(self):
        if "start_scheduling_service" in self and self['start_scheduling_service'] is False:
            return False
        else:
            return True
