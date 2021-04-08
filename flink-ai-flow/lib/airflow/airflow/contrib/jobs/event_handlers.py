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

from typing import Tuple, List, Dict, Text
import json
import time
import copy
from enum import Enum
from airflow.executors.scheduling_action import SchedulingAction
from airflow.models.eventhandler import EventHandler
from notification_service.base_notification import BaseEvent, UNDEFINED_EVENT_TYPE


class StartEventHandler(EventHandler):
    """
    Internal event handler that always return start scheduling action.
    """

    def handle_event(self, event: BaseEvent, task_state: object) -> Tuple[SchedulingAction, object]:
        return SchedulingAction.START, task_state


class RestartEventHandler(EventHandler):
    """
    Internal event handler that always return restart scheduling action.
    """

    def handle_event(self, event: BaseEvent, task_state: object) -> Tuple[SchedulingAction, object]:
        return SchedulingAction.RESTART, task_state


class ActionEventHandler(EventHandler):
    """
    Internal event handler that handle event base on the key of the event.
    """

    def handle_event(self, event: BaseEvent, task_state: object) -> Tuple[SchedulingAction, object]:
        if 'stop' == event.key:
            return SchedulingAction.STOP, task_state
        elif 'restart' == event.key:
            return SchedulingAction.RESTART, task_state
        else:
            return SchedulingAction.START, task_state


class MetCondition(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    NECESSARY = "NECESSARY"


class EventLife(str, Enum):
    """
    ONCE: the event value will be used only once
    REPEATED: the event value will be used repeated
    """
    ONCE = "ONCE"
    REPEATED = "REPEATED"


class MetValueCondition(str, Enum):
    """
    EQUAL: the condition that notification service updates a value which equals to the event value
    UPDATE: the condition that notification service has a update operation on the event key which event
            value belongs to
    """
    EQUAL = "EQUAL"
    UPDATE = "UPDATE"


DEFAULT_NAMESPACE = 'default'


class MetConfig(object):
    def __init__(self,
                 event_key: Text,
                 event_value: Text,
                 event_type: Text = UNDEFINED_EVENT_TYPE,
                 condition: MetCondition = MetCondition.NECESSARY,
                 action: SchedulingAction = SchedulingAction.START,
                 life: EventLife = EventLife.ONCE,
                 value_condition: MetValueCondition = MetValueCondition.EQUAL,
                 namespace: Text = DEFAULT_NAMESPACE
                 ):
        self.event_type = event_type
        self.event_key = event_key
        self.event_value = event_value
        self.condition = condition
        self.action = action
        self.life = life
        self.value_condition = value_condition
        self.namespace = namespace


class AiFlowTs(object):
    def __init__(self):
        self.event_map = {}
        self.schedule_time = 0

    def __str__(self):
        return "EVENT_MAP: {0} TIME: {1}".format(self.event_map, self.schedule_time)


class ActionWrapper(object):
    def __init__(self, action=SchedulingAction.NONE):
        self.action = action


class AIFlowHandler(EventHandler):
    def __init__(self, config: str):
        self.config = config

    @staticmethod
    def parse_configs(config_str: str):
        configs: List[MetConfig] = []
        config_json = json.loads(config_str)
        for config in config_json:
            met_config = MetConfig(event_key=config['event_key'],
                                   event_value=config['event_value'],
                                   event_type=config['event_type'],
                                   action=SchedulingAction(config['action']),
                                   condition=MetCondition(config['condition']),
                                   value_condition=MetValueCondition(config['value_condition']),
                                   life=EventLife(config['life']),
                                   namespace=config['namespace'])
            configs.append(met_config)
        return configs

    def handle_event(self, event: BaseEvent, task_state: object) -> Tuple[SchedulingAction, object]:
        configs: List[MetConfig] = AIFlowHandler.parse_configs(self.config)
        if task_state is None:
            task_state = AiFlowTs()
        af_ts = copy.deepcopy(task_state)
        af_ts.event_map[(event.namespace, event.key, event.event_type)] = event
        aw = ActionWrapper()
        res = self.met_sc(configs, af_ts, aw)
        if res:
            if aw.action in SchedulingAction:
                af_ts.schedule_time = time.time_ns()
            if len(configs) == 0:
                return SchedulingAction.START, af_ts
            else:
                return aw.action, af_ts
        else:
            return SchedulingAction.NONE, af_ts

    def met_sc(self, configs, ts: AiFlowTs, aw: ActionWrapper)->bool:
        event_map: Dict = ts.event_map
        schedule_time = ts.schedule_time
        has_necessary_edge = False
        for met_config in configs:
            namespace = met_config.namespace
            key = met_config.event_key
            value = met_config.event_value
            event_type = met_config.event_type

            if met_config.condition == MetCondition.SUFFICIENT:
                if (namespace, key, event_type) not in event_map:
                    continue
                event: BaseEvent = event_map[(namespace, key, event_type)]
                v, event_time = event.value, event.create_time
                if met_config.life == EventLife.ONCE:
                    if met_config.value_condition == MetValueCondition.EQUAL:
                        if v == value and event_time > schedule_time:
                            aw.action = met_config.action
                            return True
                    else:
                        if event_time > schedule_time:
                            aw.action = met_config.action
                            return True
                else:
                    if met_config.value_condition == MetValueCondition.EQUAL:
                        if v == value:
                            aw.action = met_config.action
                            return True
                    else:
                        aw.action = met_config.action
                        return True
            else:
                has_necessary_edge = True
                if (namespace, key, event_type) not in event_map:
                    return False
                event: BaseEvent = event_map[(namespace, key, event_type)]
                v, event_time = event.value, event.create_time
                if met_config.life == EventLife.ONCE:
                    if met_config.value_condition == MetValueCondition.EQUAL:
                        if schedule_time >= event_time:
                            return False
                        else:
                            if v != value:
                                return False
                    else:
                        if schedule_time >= event_time:
                            return False

                else:
                    if met_config.value_condition == MetValueCondition.EQUAL:
                        if v != value:
                            return False

                    else:
                        if v is None:
                            return False
        if has_necessary_edge:
            aw.action = configs[0].action
            return True
        else:
            return False
