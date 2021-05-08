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
import threading
from random import shuffle

import grpc
import logging
import time
from functools import wraps

from ai_flow.project.project_config import ProjectConfig
from ai_flow.rest_endpoint.protobuf.deploy_service_pb2_grpc import DeployServiceStub
from ai_flow.rest_endpoint.protobuf.high_availability_pb2 import ListMembersRequest, ReturnStatus
from ai_flow.rest_endpoint.protobuf.high_availability_pb2_grpc import HighAvailabilityManagerStub
from ai_flow.rest_endpoint.protobuf.metadata_service_pb2_grpc import MetadataServiceStub
from ai_flow.rest_endpoint.protobuf.metric_service_pb2_grpc import MetricServiceStub
from ai_flow.rest_endpoint.protobuf.model_center_service_pb2_grpc import ModelCenterServiceStub
from ai_flow.rest_endpoint.protobuf.scheduling_service_pb2_grpc import SchedulingServiceStub
from ai_flow.rest_endpoint.service.high_availability import proto_to_member, sleep_and_detecting_running
from notification_service.base_notification import BaseEvent
from ai_flow.notification.event_types import AI_FLOW_TYPE
from ai_flow.rest_endpoint.service.client.deploy_client import DeployClient
from ai_flow.rest_endpoint.service.client.metadata_client import MetadataClient
from ai_flow.rest_endpoint.service.client.model_center_client import ModelCenterClient
from notification_service.client import NotificationClient
from ai_flow.rest_endpoint.service.client.metric_client import MetricClient
from ai_flow.rest_endpoint.service.client.scheduling_client import SchedulingClient


if not hasattr(time, 'time_ns'):
    time.time_ns = lambda: int(time.time() * 1e9)

_SERVER_URI = 'localhost:50051'


class AIFlowClient(MetadataClient, ModelCenterClient, NotificationClient, DeployClient, MetricClient, SchedulingClient):
    """
    Client of an AIFlow Server that manages metadata store, model center and notification service.
    """

    def __init__(self,
                 server_uri=_SERVER_URI,
                 notification_service_uri=None,
                 project_config: ProjectConfig = None):
        MetadataClient.__init__(self, server_uri)
        ModelCenterClient.__init__(self, server_uri)
        DeployClient.__init__(self, server_uri)
        MetricClient.__init__(self, server_uri)
        SchedulingClient.__init__(self, server_uri)
        self.enable_ha = False
        self.list_member_interval_ms = 5000
        self.retry_interval_ms = 1000
        self.retry_timeout_ms = 10000
        if project_config is not None:
            if server_uri is None:
                server_uri = project_config.get_master_uri()
            if notification_service_uri is None:
                notification_service_uri = project_config.get_notification_service_uri()
            self.enable_ha = project_config.get_enable_ha()
            self.list_member_interval_ms = project_config.get_list_member_interval_ms()
            self.retry_interval_ms = project_config.get_retry_interval_ms()
            self.retry_timeout_ms = project_config.get_retry_timeout_ms()
        if notification_service_uri is None:
            NotificationClient.__init__(
                self,
                server_uri,
                enable_ha=self.enable_ha,
                list_member_interval_ms=self.list_member_interval_ms,
                retry_interval_ms=self.retry_interval_ms,
                retry_timeout_ms=self.retry_timeout_ms)
        else:
            NotificationClient.__init__(
                self,
                notification_service_uri,
                enable_ha=self.enable_ha,
                list_member_interval_ms=self.list_member_interval_ms,
                retry_interval_ms=self.retry_interval_ms,
                retry_timeout_ms=self.retry_timeout_ms)
        if self.enable_ha:
            server_uris = server_uri.split(",")
            self.living_aiflow_members = []
            self.current_aiflow_uri = None
            last_error = None
            for server_uri in server_uris:
                channel = grpc.insecure_channel(server_uri)
                high_availability_stub = HighAvailabilityManagerStub(channel)
                try:
                    request = ListMembersRequest(timeout_seconds=0)
                    response = high_availability_stub.listMembers(request)
                    if response.return_code == ReturnStatus.CALL_SUCCESS:
                        self.living_aiflow_members = [proto_to_member(proto).server_uri
                                                      for proto in response.members]
                    else:
                        raise Exception(response.return_msg)
                    self.current_aiflow_uri = server_uri
                    self.high_availability_stub = high_availability_stub
                    break
                except grpc.RpcError as e:
                    last_error = e
            if self.current_aiflow_uri is None:
                raise Exception("No available aiflow server uri!") from last_error
            self.aiflow_ha_change_lock = threading.Lock()
            self.aiflow_ha_running = True
            self._replace_aiflow_stubs(self.current_aiflow_uri)
            self.list_aiflow_member_thread = threading.Thread(target=self._list_aiflow_members, daemon=True)
            self.list_aiflow_member_thread.start()

    def publish_event(self, key: str, value: str, event_type: str = AI_FLOW_TYPE) -> BaseEvent:
        return self.send_event(BaseEvent(key, value, event_type))

    def _list_aiflow_members(self):
        while self.aiflow_ha_running:
            # refresh the living members
            request = ListMembersRequest(timeout_seconds=int(self.list_member_interval_ms / 1000))
            response = self.high_availability_stub.listMembers(request)
            if response.return_code == ReturnStatus.CALL_SUCCESS:
                with self.aiflow_ha_change_lock:
                    self.living_aiflow_members = [proto_to_member(proto).server_uri
                                                  for proto in response.members]
            else:
                logging.error("Exception thrown when updating the living members: %s" %
                              response.return_msg)

    def _aiflow_ha_wrapper(self, func, stub_name):
        @wraps(func)
        def call_with_retry(*args, **kwargs):
            current_stub = getattr(self, stub_name)
            current_func = getattr(current_stub, func.__name__).inner_func
            start_time = time.time_ns() / 1000000
            failed_members = set()
            while True:
                try:
                    return current_func(*args, **kwargs)
                except grpc.RpcError:
                    logging.error("Exception thrown when calling rpc, change the connection.",
                                  exc_info=True)
                    with self.aiflow_ha_change_lock:
                        # check the current_uri to ensure thread safety
                        if current_func.server_uri == self.current_aiflow_uri:
                            living_members = list(self.living_aiflow_members)
                            failed_members.add(self.current_aiflow_uri)
                            shuffle(living_members)
                            found_new_member = False
                            for server_uri in living_members:
                                if server_uri in failed_members:
                                    continue
                                next_uri = server_uri

                                self._replace_aiflow_stubs(next_uri)

                                current_func = getattr(getattr(self, stub_name),
                                                       current_func.__name__).inner_func
                                self.current_aiflow_uri = next_uri
                                found_new_member = True
                            if not found_new_member:
                                logging.error("No available living members currently. Sleep and retry.")
                                failed_members.clear()
                                sleep_and_detecting_running(self.retry_interval_ms,
                                                            lambda: self.aiflow_ha_running)

                # break if stopped or timeout
                if not self.aiflow_ha_running or \
                        time.time_ns() / 1000000 > start_time + self.retry_timeout_ms:
                    if not self.aiflow_ha_running:
                        raise Exception("HA has been disabled.")
                    else:
                        raise Exception("Rpc retry timeout!")

        call_with_retry.inner_func = func
        return call_with_retry

    def _wrap_aiflow_rpcs(self, stub, server_uri, stub_name):
        for method_name, method in dict(stub.__dict__).items():
            method.__name__ = method_name
            method.server_uri = server_uri
            setattr(stub, method_name, self._aiflow_ha_wrapper(method, stub_name))
        return stub

    def _replace_aiflow_stubs(self, server_uri):
        high_availability_channel = grpc.insecure_channel(server_uri)
        high_availability_stub = self._wrap_aiflow_rpcs(
            HighAvailabilityManagerStub(high_availability_channel),
            server_uri,
            "high_availability_stub")
        self.high_availability_stub = high_availability_stub

        metadata_channel = grpc.insecure_channel(server_uri)
        metadata_store_stub = self._wrap_aiflow_rpcs(
            MetadataServiceStub(metadata_channel),
            server_uri,
            "metadata_store_stub")
        self.metadata_store_stub = metadata_store_stub

        model_center_channel = grpc.insecure_channel(server_uri)
        model_center_stub = self._wrap_aiflow_rpcs(
            ModelCenterServiceStub(model_center_channel),
            server_uri,
            "model_center_stub")
        self.model_center_stub = model_center_stub

        deploy_channel = grpc.insecure_channel(server_uri)
        deploy_stub = self._wrap_aiflow_rpcs(
            DeployServiceStub(deploy_channel),
            server_uri,
            "deploy_stub")
        self.deploy_stub = deploy_stub

        metric_channel = grpc.insecure_channel(server_uri)
        metric_stub = self._wrap_aiflow_rpcs(
            MetricServiceStub(metric_channel),
            server_uri,
            "metric_stub")
        self.metric_stub = metric_stub

        scheduling_channel = grpc.insecure_channel(server_uri)
        scheduling_stub = self._wrap_aiflow_rpcs(
            SchedulingServiceStub(scheduling_channel),
            server_uri,
            "scheduling_stub")
        self.scheduling_stub = scheduling_stub

    def disable_high_availability(self):
        if hasattr(self, "aiflow_ha_running"):
            self.aiflow_ha_running = False
        NotificationClient.disable_high_availability(self)
        if hasattr(self, "aiflow_ha_running"):
            self.list_aiflow_member_thread.join()
