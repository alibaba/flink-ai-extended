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
import json

from notification_service.base_notification import BaseEvent
from notification_service.client import NotificationClient

from ai_flow.common.constants import DEFAULT_NAMESPACE
from ai_flow.model_center.entity.model_version import ModelVersion
from ai_flow.model_center.entity.model_version_param import ModelVersionParam
from ai_flow.model_center.entity.model_version_stage import MODEL_VERSION_TO_EVENT_TYPE, ModelVersionEventType, \
    ModelVersionStage
from ai_flow.model_center.entity.registered_model import RegisteredModel
from ai_flow.model_center.entity.registered_model_param import RegisteredModelParam
from ai_flow.rest_endpoint.protobuf import model_center_service_pb2_grpc
from ai_flow.rest_endpoint.protobuf.message_pb2 import ModelType, RegisteredModelMetas
from ai_flow.rest_endpoint.service.util import catch_exception, _wrap_response
from ai_flow.store.sqlalchemy_store import SqlAlchemyStore
from ai_flow.store.mongo_store import MongoStore
from ai_flow.store.db.db_util import extract_db_engine_from_uri, parse_mongo_uri
from ai_flow.application_master.master_config import DBType


class ModelCenterService(model_center_service_pb2_grpc.ModelCenterServiceServicer):

    def __init__(self, store_uri, server_uri, notification_uri=None):
        db_engine = extract_db_engine_from_uri(store_uri)
        if DBType.value_of(db_engine) == DBType.MONGODB:
            username, password, host, port, db = parse_mongo_uri(store_uri)
            self.model_repo_store = MongoStore(host=host,
                                               port=int(port),
                                               username=username,
                                               password=password,
                                               db=db)
        else:
            self.model_repo_store = SqlAlchemyStore(store_uri)
        if notification_uri is None:
            self.notification_client = NotificationClient(server_uri)
        else:
            self.notification_client = NotificationClient(notification_uri)

    @catch_exception
    def createRegisteredModel(self, request, context):
        registered_model_param = RegisteredModelParam.from_proto(request)
        registered_model_meta = self.model_repo_store.create_registered_model(registered_model_param.model_name,
                                                                              ModelType.Name(
                                                                                  registered_model_param.model_type),
                                                                              registered_model_param.model_desc)
        return _wrap_response(registered_model_meta.to_meta_proto())

    @catch_exception
    def updateRegisteredModel(self, request, context):
        model_meta_param = RegisteredModel.from_proto(request)
        registered_model_param = RegisteredModelParam.from_proto(request)
        registered_model_meta = self.model_repo_store.update_registered_model(
            RegisteredModel(model_meta_param.model_name),
            registered_model_param.model_name,
            ModelType.Name(
                registered_model_param.model_type),
            registered_model_param.model_desc)
        return _wrap_response(None if registered_model_meta is None else registered_model_meta.to_meta_proto())

    @catch_exception
    def deleteRegisteredModel(self, request, context):
        model_meta_param = RegisteredModel.from_proto(request)
        self.model_repo_store.delete_registered_model(RegisteredModel(model_name=model_meta_param.model_name))
        return _wrap_response(request.model_meta)

    @catch_exception
    def listRegisteredModels(self, request, context):
        registered_models = self.model_repo_store.list_registered_models()
        return _wrap_response(RegisteredModelMetas(registered_models=[registered_model.to_meta_proto()
                                                                      for registered_model in registered_models]))

    @catch_exception
    def getRegisteredModelDetail(self, request, context):
        model_meta_param = ModelVersion.from_proto(request)
        registered_model_detail = self.model_repo_store.get_registered_model_detail(
            RegisteredModel(model_name=model_meta_param.model_name))
        return _wrap_response(None if registered_model_detail is None else registered_model_detail.to_detail_proto())

    @catch_exception
    def createModelVersion(self, request, context):
        model_meta_param = ModelVersion.from_proto(request)
        model_version_param = ModelVersionParam.from_proto(request)
        model_version_meta = self.model_repo_store.create_model_version(model_meta_param.model_name,
                                                                        model_version_param.model_path,
                                                                        model_version_param.model_metric,
                                                                        model_version_param.model_flavor,
                                                                        model_version_param.version_desc,
                                                                        model_version_param.current_stage)
        model_type = MODEL_VERSION_TO_EVENT_TYPE.get(ModelVersionStage.from_string(model_version_param.current_stage))
        self.notification_client.send_event(BaseEvent(model_version_meta.model_name,
                                                      json.dumps(model_version_meta.__dict__),
                                                      model_type,
                                                      namespace=DEFAULT_NAMESPACE))
        return _wrap_response(model_version_meta.to_meta_proto())

    @catch_exception
    def updateModelVersion(self, request, context):
        model_meta_param = ModelVersion.from_proto(request)
        model_version_param = ModelVersionParam.from_proto(request)
        model_version_meta = self.model_repo_store.update_model_version(model_meta_param,
                                                                        model_version_param.model_path,
                                                                        model_version_param.model_metric,
                                                                        model_version_param.model_flavor,
                                                                        model_version_param.version_desc,
                                                                        model_version_param.current_stage)
        if model_version_param.current_stage is not None:
            model_type = MODEL_VERSION_TO_EVENT_TYPE.get(ModelVersionStage.from_string(model_version_param.current_stage))
            self.notification_client.send_event(BaseEvent(model_version_meta.model_name,
                                                          json.dumps(model_version_meta.__dict__),
                                                          model_type,
                                                          namespace=DEFAULT_NAMESPACE))
        return _wrap_response(None if model_version_meta is None else model_version_meta.to_meta_proto())

    @catch_exception
    def deleteModelVersion(self, request, context):
        model_meta_param = ModelVersion.from_proto(request)
        self.model_repo_store.delete_model_version(model_meta_param)
        self.notification_client.send_event(BaseEvent(model_meta_param.model_name,
                                                      json.dumps(model_meta_param.__dict__),
                                                      ModelVersionEventType.MODEL_DELETED))
        return _wrap_response(request.model_meta)

    @catch_exception
    def getModelVersionDetail(self, request, context):
        model_meta_param = ModelVersion.from_proto(request)
        model_version_meta = self.model_repo_store.get_model_version_detail(model_meta_param)
        return _wrap_response(None if model_version_meta is None else model_version_meta.to_meta_proto())
