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
from typing import Text
from ai_flow.rest_endpoint.service.server import AIFlowServer, HighAvailableAIFlowServer
from ai_flow.store.db.base_model import base
from ai_flow.store.sqlalchemy_store import SqlAlchemyStore
from ai_flow.store.mongo_store import MongoStoreConnManager
from ai_flow.application_master.master_config import MasterConfig, DBType
from ai_flow.client.ai_flow_client import get_ai_flow_client
import logging

_SQLITE_DB_FILE = 'aiflow.db'
_SQLITE_DB_URI = '%s%s' % ('sqlite:///', _SQLITE_DB_FILE)
_MYSQL_DB_URI = 'mysql+pymysql://root:aliyunmysql@localhost:3306/aiflow'
_PORT = '50051'
GLOBAL_MASTER_CONFIG = {}


class AIFlowMaster(object):
    """
    AI flow master.
    """
    def __init__(self, config_file: Text = None, enable_ha=False, server_uri: str = None, ttl_ms=10000) -> None:
        """
        Set the master attribute according to the master config file.

        :param config_file: master configuration file.
        """
        super().__init__()
        self.config_file = config_file
        self.server = None
        self.master_config = MasterConfig()
        self.enable_ha = enable_ha
        self.server_uri = server_uri
        self.ttl_ms = ttl_ms

    def start(self,
              is_block=False) -> None:
        """
        Start the AI flow master.

        :param is_block: AI flow master will run non-stop if True.
        """
        if self.config_file is not None:
            self.master_config.load_from_file(self.config_file)
        else:
            self.master_config.set_master_port(str(_PORT))
        global GLOBAL_MASTER_CONFIG
        GLOBAL_MASTER_CONFIG = self.master_config
        logging.info("AI Flow Master Config {}".format(GLOBAL_MASTER_CONFIG))
        if not self.master_config.get_enable_ha():
            self.server = AIFlowServer(
                store_uri=self.master_config.get_db_uri(),
                port=str(self.master_config.get_master_port()),
                start_default_notification=self.master_config.start_default_notification(),
                notification_uri=self.master_config.get_notification_uri(),
                start_meta_service=self.master_config.start_meta_service(),
                start_model_center_service=self.master_config.start_model_center_service(),
                start_metric_service=self.master_config.start_metric_service(),
                start_deploy_service=self.master_config.start_deploy_service(),
                start_scheduling_service=self.master_config.start_scheduling_service(),
                scheduler_config=self.master_config.get_scheduler_config())
        else:
            self.server = HighAvailableAIFlowServer(
                store_uri=self.master_config.get_db_uri(),
                port=str(self.master_config.get_master_port()),
                start_default_notification=self.master_config.start_default_notification(),
                notification_uri=self.master_config.get_notification_uri(),
                server_uri=self.master_config.get_master_ip() + ":" + str(self.master_config.get_master_port()),
                ttl_ms=self.master_config.get_ha_ttl_ms())
        self.server.run(is_block=is_block)

    def stop(self, clear_sql_lite_db_file=True) -> None:
        """
        Stop the AI flow master.

        :param clear_sql_lite_db_file: If True, the sqlite database files will be deleted When the server stops working.
        """
        self.server.stop()
        if self.master_config.get_db_type() == DBType.SQLITE and clear_sql_lite_db_file:
            store = SqlAlchemyStore(self.master_config.get_db_uri())
            base.metadata.drop_all(store.db_engine)
            os.remove(self.master_config.get_sql_lite_db_file())
        elif self.master_config.get_db_type() == DBType.MONGODB:
            MongoStoreConnManager().disconnect_all()

    def _clear_db(self):
        if self.master_config.get_db_type() == DBType.SQLITE:
            store = SqlAlchemyStore(self.master_config.get_db_uri())
            base.metadata.drop_all(store.db_engine)
            base.metadata.create_all(store.db_engine)
        elif self.master_config.get_db_type() == DBType.MONGODB:
            MongoStoreConnManager().drop_all()


def set_master_config():
    code, config, message = get_ai_flow_client().get_master_config()
    for k, v in config.items():
        GLOBAL_MASTER_CONFIG[k] = v
