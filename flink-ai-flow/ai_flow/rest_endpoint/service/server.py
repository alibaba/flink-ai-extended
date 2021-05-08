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
import asyncio
import functools
import inspect
import os
import sys
import tempfile
import threading
import time
from concurrent import futures
import logging
import grpc
from grpc import _common, _server
from grpc._cython.cygrpc import StatusCode
from grpc._server import _serialize_response, _status, _abort, _Context, _unary_request, \
    _select_thread_pool_for_behavior, _unary_response_in_pool
from typing import Dict

from ai_flow.rest_endpoint.protobuf.high_availability_pb2_grpc import add_HighAvailabilityManagerServicer_to_server
from ai_flow.rest_endpoint.service.high_availability import SimpleAIFlowServerHaManager, HighAvailableService
from ai_flow.store.sqlalchemy_store import SqlAlchemyStore
from ai_flow.store.mongo_store import MongoStore
from ai_flow.store.db.db_util import extract_db_engine_from_uri, parse_mongo_uri
from ai_flow.application_master.master_config import DBType
from notification_service.proto import notification_service_pb2_grpc

from ai_flow.deployer.deploy_service import DeployService
from ai_flow.metadata_store.service.service import MetadataService
from ai_flow.model_center.service.service import ModelCenterService
from ai_flow.notification.service.service import NotificationService
from ai_flow.metric.service.metric_service import MetricService
from ai_flow.scheduler.scheduling_service import SchedulingService, SchedulerConfig


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))

from ai_flow.rest_endpoint.protobuf import model_center_service_pb2_grpc, \
    metadata_service_pb2_grpc, deploy_service_pb2_grpc, metric_service_pb2_grpc, scheduling_service_pb2_grpc

_PORT = '50051'
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class AIFlowServer(object):
    """
    Block/Async server of an AIFlow Rest Endpoint that provides Metadata/Model/Notification function service.
    """

    def __init__(self, store_uri=None, port=_PORT,
                 start_default_notification: bool = True,
                 notification_uri=None,
                 start_meta_service: bool = True,
                 start_model_center_service: bool = True,
                 start_metric_service: bool = True,
                 start_deploy_service: bool = True,
                 start_scheduling_service: bool = True,
                 scheduler_config: Dict = None):
        self.executor = Executor(futures.ThreadPoolExecutor(max_workers=10))
        self.server = grpc.server(self.executor)
        self.start_default_notification = start_default_notification
        server_uri = 'localhost:{}'.format(port)
        if start_default_notification:
            logging.info("start default notification service.")
            notification_service_pb2_grpc.add_NotificationServiceServicer_to_server(NotificationService(store_uri),
                                                                                    self.server)
        if start_model_center_service:
            logging.info("start model center service.")
            model_center_service_pb2_grpc.add_ModelCenterServiceServicer_to_server(
                ModelCenterService(store_uri=store_uri, server_uri=server_uri,
                                   notification_uri=notification_uri), self.server)
        if start_meta_service:
            logging.info("start meta service.")
            metadata_service_pb2_grpc.add_MetadataServiceServicer_to_server(
                MetadataService(db_uri=store_uri, server_uri=server_uri), self.server)
        if start_metric_service:
            logging.info("start metric service.")
            metric_service_pb2_grpc.add_MetricServiceServicer_to_server(MetricService(db_uri=store_uri), self.server)

        if start_deploy_service:
            logging.info("start deploy service.")
            self.start_deploy_service = True
            self.deploy_service = DeployService(server_uri=server_uri)
            deploy_service_pb2_grpc.add_DeployServiceServicer_to_server(self.deploy_service, self.server)
        else:
            self.start_deploy_service = False

        if start_scheduling_service:
            logging.info("start scheduling service.")
            if scheduler_config is None:
                nf_uri = server_uri if start_default_notification else notification_uri
                scheduler_config = SchedulerConfig()
                scheduler_config.set_notification_service_uri(nf_uri)
                scheduler_config.\
                    set_scheduler_class_name('ai_flow.scheduler.implements.airflow_scheduler.AirFlowScheduler')
                scheduler_config.set_repository('/tmp/airflow')
            real_config = SchedulerConfig()
            if scheduler_config.get('notification_uri') is None:
                nf_uri = server_uri if start_default_notification else notification_uri
                real_config.set_notification_service_uri(nf_uri)
            else:
                real_config.set_notification_service_uri(scheduler_config.get('notification_uri'))
            real_config.set_properties(scheduler_config.get('properties'))
            real_config.set_repository(scheduler_config.get('repository'))
            real_config.set_scheduler_class_name(scheduler_config.get('scheduler_class_name'))
            self.scheduling_service = SchedulingService(real_config)
            scheduling_service_pb2_grpc.add_SchedulingServiceServicer_to_server(self.scheduling_service,
                                                                                self.server)

        self.server.add_insecure_port('[::]:' + str(port))

    def run(self, is_block=False):
        self.server.start()
        logging.info('AIFlow server started.')
        if self.start_deploy_service:
            self.deploy_service.start_scheduler_manager()
            logging.info('Deploy service started.')
        if is_block:
            try:
                while True:
                    time.sleep(_ONE_DAY_IN_SECONDS)
            except KeyboardInterrupt:
                self.stop()
        else:
            pass

    def stop(self):
        if self.start_deploy_service:
            self.deploy_service.stop_scheduler_manager()
        self.executor.shutdown()
        self.server.stop(0)
        logging.info('AIFlow server stopped.')


class HighAvailableAIFlowServer(AIFlowServer):
    """
    High available AIFlow server.
    """

    def __init__(self,
                 store_uri=None,
                 port=_PORT,
                 start_default_notification: bool = True,
                 notification_uri=None,
                 ha_manager=None,
                 server_uri=None,
                 ha_storage=None,
                 ttl_ms: int = 10000):
        super(HighAvailableAIFlowServer, self).__init__(
            store_uri, port, start_default_notification, notification_uri)
        if ha_manager is None:
            ha_manager = SimpleAIFlowServerHaManager()
        if server_uri is None:
            raise ValueError("server_uri is required!")
        if ha_storage is None:
            db_engine = extract_db_engine_from_uri(store_uri)
            if DBType.value_of(db_engine) == DBType.MONGODB:
                username, password, host, port, db = parse_mongo_uri(store_uri)
                ha_storage = MongoStore(host=host,
                                        port=int(port),
                                        username=username,
                                        password=password,
                                        db=db)
            else:
                ha_storage = SqlAlchemyStore(store_uri)
        self.ha_service = HighAvailableService(ha_manager, server_uri, ha_storage, ttl_ms)
        add_HighAvailabilityManagerServicer_to_server(self.ha_service, self.server)

    def run(self, is_block=False):
        self.ha_service.start()
        logging.info('HA service started.')
        super(HighAvailableAIFlowServer, self).run(is_block)

    def stop(self):
        super(HighAvailableAIFlowServer, self).stop()
        logging.info('HA service stopped.')
        self.ha_service.stop()


def _loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    if not loop.is_running() or loop.is_closed():
        loop.run_forever()
    pending = asyncio.all_tasks(loop=loop)
    if pending:
        loop.run_until_complete(asyncio.gather(*pending))


class Executor(futures.Executor):
    def __init__(self, thread_pool, loop=None):
        super().__init__()
        self._shutdown = False
        self._thread_pool = thread_pool
        self._loop = loop or asyncio.get_event_loop()
        if not self._loop.is_running() or self._loop.is_closed():
            self._thread = threading.Thread(target=_loop, args=(self._loop,), daemon=True)
            self._thread.start()

    def submit(self, fn, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError('Cannot schedule new futures after shutdown.')
        if not self._loop.is_running():
            raise RuntimeError('Loop must be started before any function could be submitted.')
        if inspect.iscoroutinefunction(fn):
            coroutine = fn(*args, **kwargs)
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        else:
            func = functools.partial(fn, *args, **kwargs)
            return self._loop.run_in_executor(self._thread_pool, func)

    def shutdown(self, wait=True):
        self._shutdown = True
        if wait:
            self._thread_pool.shutdown()


async def _call_behavior_async(rpc_event, state, behavior, argument, request_deserializer):
    context = _Context(rpc_event, state, request_deserializer)
    try:
        return await behavior(argument, context), True
    except Exception as e:
        with state.condition:
            if e not in state.rpc_errors:
                logging.exception(e)
                _abort(state, rpc_event.operation_call, StatusCode.unknown, _common.encode(e))
        return None, False


async def _unary_response_in_pool_async(rpc_event, state, behavior, argument_thunk, request_deserializer,
                                        response_serializer):
    argument = argument_thunk()
    if argument is not None:
        response, proceed = await _call_behavior_async(rpc_event, state, behavior, argument, request_deserializer)
        if proceed:
            serialized_response = _serialize_response(rpc_event, state, response, response_serializer)
            if serialized_response is not None:
                _status(rpc_event, state, serialized_response)


def _handle_unary_unary(rpc_event, state, method_handler, default_thread_pool):
    unary_request = _unary_request(rpc_event, state, method_handler.request_deserializer)
    thread_pool = _select_thread_pool_for_behavior(method_handler.unary_unary, default_thread_pool)
    if asyncio.iscoroutinefunction(method_handler.unary_unary):
        return thread_pool.submit(_unary_response_in_pool_async, rpc_event, state, method_handler.unary_unary,
                                  unary_request, method_handler.request_deserializer,
                                  method_handler.response_serializer)
    else:
        return thread_pool.submit(_unary_response_in_pool, rpc_event, state, method_handler.unary_unary, unary_request,
                                  method_handler.request_deserializer, method_handler.response_serializer)


_server._handle_unary_unary = _handle_unary_unary

if __name__ == '__main__':
    fd, temp_db_file = tempfile.mkstemp()
    os.close(fd)
    store_uri = '%s%s' % ('sqlite:///', temp_db_file)
    server = AIFlowServer(store_uri=store_uri)
    server.run(is_block=True)
