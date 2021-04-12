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
from notification_service.base_notification import EventWatcher, BaseEvent, UNDEFINED_EVENT_TYPE
from ai_flow.client.ai_flow_client import get_ai_flow_client
from ai_flow.common.constants import DEFAULT_NAMESPACE


def send_event(key: str, value: str, event_type: str = UNDEFINED_EVENT_TYPE, namespace: str = DEFAULT_NAMESPACE):
    """
    Send event to Notification Service.

    :param event_type: the event type.
    :param key: Key of events updated in Notification Service.
    :param value: Value of events updated in Notification Service.
    :param namespace: Namespace of the event in Notification Service, used to control the visibility of the event.
    :return: A single object of :py:class:`ai_flow.notification.base_notification.Event`
    created in notification service.
    """
    return get_ai_flow_client().send_event(BaseEvent(key=key, value=value, event_type=event_type, namespace=namespace))


def list_events(key: str, version: int = None) -> list:
    """
    List specific `key` or `version` events in Notification Service.

    :param key: Key of event for listening.
    :param version: (Optional) Version of event for listening.
    """
    return get_ai_flow_client().list_events(key=key, version=version)


def start_listen_event(key: str,
                       watcher: EventWatcher,
                       namespace: str = None,
                       version: int = None,
                       event_type: str = None,
                       start_time: int = None):
    """
    Start listen specific `key` or `version` events in Notification Service.

    :param key: Key of notification for listening.
    :param watcher: Watcher instance for listening.
    :param namespace: Namespace of the event for listening.
    :param version: (Optional) The version of the events must greater than this version.
    :param event_type: (Optional) Type of the events for listening.
    :param start_time: (Optional) Start time of the events for listening.
    """
    get_ai_flow_client().start_listen_event(key=key,
                                            watcher=watcher,
                                            namespace=namespace,
                                            version=version,
                                            event_type=event_type,
                                            start_time=start_time)


def stop_listen_event(key: str = None):
    """
    Stop listen specific `key` events in Notification Service.
    :param key: Key of event for listening.
    """
    get_ai_flow_client().stop_listen_event(key=key)
