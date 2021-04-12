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
from enum import Enum
from typing import Text

from ai_flow.common.json_utils import Jsonable
from notification_service.base_notification import UNDEFINED_EVENT_TYPE
from ai_flow.common.constants import DEFAULT_NAMESPACE


class Edge(Jsonable):
    """ the edge connect tow node"""

    def __init__(self,
                 target_node_id: Text,
                 source_node_id: Text,
                 ) -> None:
        """

        :param target_node_id: the node dependent the other node output id
        :param source_node_id: the node id
        """
        super().__init__()
        if target_node_id is None or source_node_id is None:
            raise Exception('target node id or source node id can not be None!')
        self.target_node_id = target_node_id
        self.source_node_id = source_node_id

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Edge):
            return self.source_node_id == o.source_node_id and self.target_node_id == o.target_node_id
        else:
            return False

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)


class DataEdge(Edge):
    def __init__(self,
                 target_node_id: Text,
                 source_node_id: Text,
                 port: int = 0,
                 data_config: Jsonable = None) -> None:
        super().__init__(source_node_id=source_node_id, target_node_id=target_node_id)
        self.port = port
        self.data_config = data_config

    def __eq__(self, o: object) -> bool:
        if isinstance(o, DataEdge):
            return self.source_node_id == o.source_node_id \
                   and self.target_node_id == o.target_node_id \
                   and self.port == o.port
        else:
            return False

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)


class MetCondition(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    NECESSARY = "NECESSARY"


class TaskAction(str, Enum):
    """
    START: start init job
    RESTART: start init job or stop running job and start the job
    """
    START = "START"
    RESTART = "RESTART"
    STOP = "STOP"


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




class MetConfig(Jsonable):
    def __init__(self,
                 event_key: Text,
                 event_value: Text,
                 event_type: Text = UNDEFINED_EVENT_TYPE,
                 condition: MetCondition = MetCondition.NECESSARY,
                 action: TaskAction = TaskAction.START,
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


def generate_job_status_key(target_id) -> str:
    return str(target_id) + ".job_status"


class ControlEdge(Edge):
    def __init__(self, target_node_id: Text, source_node_id: Text, namespace=DEFAULT_NAMESPACE):
        super().__init__(target_node_id, source_node_id)
        self.namespace = namespace

    def generate_met_config(self) -> MetConfig:
        return MetConfig(event_key=generate_job_status_key(self.target_node_id),
                         event_value="FINISHED",
                         namespace=self.namespace)


class UserDefineControlEdge(ControlEdge):

    def __init__(self, target_node_id: Text,
                 source_node_id: Text,
                 event_key: Text,
                 event_value: Text,
                 event_type: Text = None,
                 condition: MetCondition = MetCondition.NECESSARY,
                 action: TaskAction = TaskAction.START,
                 life: EventLife = EventLife.ONCE,
                 value_condition: MetValueCondition = MetValueCondition.EQUAL,
                 namespace: Text = DEFAULT_NAMESPACE
                 ) -> None:
        super().__init__(target_node_id, source_node_id, namespace)
        self.event_key = event_key
        self.event_value = event_value
        self.event_type = event_type
        self.condition = condition
        self.action = action
        self.life = life
        self.value_condition = value_condition

    def generate_met_config(self) -> MetConfig:
        return MetConfig(event_key=self.event_key,
                         event_value=self.event_value,
                         event_type=self.event_type,
                         condition=self.condition,
                         action=self.action,
                         life=self.life,
                         value_condition=self.value_condition,
                         namespace=self.namespace)


class JobControlEdge(Edge):
    def __init__(self,
                 target_node_id: Text,
                 source_node_id: Text = None,
                 met_config: MetConfig = None,
                 namespace: Text = DEFAULT_NAMESPACE
                 ) -> None:
        super().__init__(target_node_id, source_node_id)
        if met_config is None:
            self.met_config = MetConfig(event_key=generate_job_status_key(target_node_id),
                                        event_value="FINISHED",
                                        namespace=namespace)
        else:
            self.met_config = met_config


def control_edge_to_job_edge(control_edge: ControlEdge) -> JobControlEdge:
    return JobControlEdge(source_node_id=control_edge.source_node_id,
                          target_node_id=control_edge.target_node_id,
                          met_config=control_edge.generate_met_config())


class StartBeforeControlEdge(ControlEdge):

    def generate_met_config(self) -> MetConfig:
        return MetConfig(event_key=generate_job_status_key(self.target_node_id),
                         event_value="STARTING", namespace=self.namespace)


class StopBeforeControlEdge(ControlEdge):
    pass


class RestartBeforeControlEdge(ControlEdge):
    def generate_met_config(self) -> MetConfig:
        return MetConfig(event_key=generate_job_status_key(self.target_node_id),
                         event_value="FINISHED",
                         namespace=self.namespace,
                         action=TaskAction.RESTART)


class ModelVersionControlEdge(ControlEdge):
    def __init__(self, model_name: Text,
                 model_type: Text,
                 target_node_id: Text,
                 source_node_id: Text = None,
                 namespace: Text = DEFAULT_NAMESPACE) -> None:
        super().__init__(target_node_id, source_node_id, namespace)
        self.model_name = model_name
        self.model_type = model_type

    def generate_met_config(self) -> MetConfig:
        return MetConfig(event_type=self.model_type,
                         event_key=self.model_name,
                         event_value="*",
                         action=TaskAction.RESTART,
                         life=EventLife.ONCE,
                         value_condition=MetValueCondition.UPDATE,
                         condition=MetCondition.SUFFICIENT,
                         namespace=self.namespace)


class ExampleControlEdge(ControlEdge):
    def __init__(self, example_name: Text,
                 target_node_id: Text,
                 source_node_id: Text = None,
                 namespace: Text = DEFAULT_NAMESPACE) -> None:
        super().__init__(target_node_id, source_node_id, namespace)
        self.example_name = example_name

    def generate_met_config(self) -> MetConfig:
        return MetConfig(event_key="example." + self.example_name, event_value="created", namespace=self.namespace)
