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
from ai_flow.common.properties import Properties


class ExecutionMode(str, Enum):
    """
    Execution mode of the specific job. Batch or Stream.
    """
    BATCH = 'BATCH'
    STREAM = 'STREAM'

    @staticmethod
    def value_of(exec_type):
        if exec_type in ('BATCH', 'batch'):
            return ExecutionMode.BATCH
        elif exec_type in ('STREAM', 'stream'):
            return ExecutionMode.STREAM
        else:
            raise NotImplementedError


class State(str, Enum):
    INIT = 'INIT'
    STARTING = 'STARTING'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    FAILED = 'FAILED'
    KILLING = 'KILLING'
    KILLED = 'KILLED'


class JobMeta(Jsonable):
    """define job meta"""

    def __init__(self,
                 name: Text,
                 workflow_execution_id: int,
                 job_state: State,
                 properties: Properties = None,
                 job_id: Text = None,
                 start_time: int = None,
                 end_time: int = None,
                 log_uri: Text = None,
                 signature: Text = None,
                 uuid: int = None
                 ) -> None:
        """
        :param name: the job name
        :param workflow_execution_id: belong to which workflow_execution
        :param properties: the job properties
        :param job_id: the generated job id
        :param start_time: the job start time
        :param end_time: the job end time
        :param job_state: the job state
        :param log_uri: the job runtime log address
        :param signature: to identify if the example is the previous one
        """
        self.name = name
        self.workflow_execution_id = workflow_execution_id
        self.properties = properties
        self.job_id = job_id
        self.start_time = start_time
        self.end_time = end_time
        self.job_state = job_state
        self.log_uri = log_uri
        self.signature = signature
        self.uuid = uuid

    def __str__(self):
        return '<\n' \
               'JobMeta\n' \
               'uuid:{},\n' \
               'name:{},\n' \
               'workflow_execution_id:{},\n' \
               'properties:{}, \n' \
               'job_id:{},\n' \
               'start_time:{},\n' \
               'end_time:{},\n' \
               'state:{},\n' \
               'log_uri:{},\n' \
               'signature: {} \n' \
               '>'.format(
            self.uuid, self.name, self.workflow_execution_id, self.properties,
            self.job_id, self.start_time, self.end_time,
            self.job_state, self.log_uri, self.signature)


def create_job(
        name: Text,
        workflow_execution_id: int,
        state: State,
        properties: Properties = None,
        job_id: Text = None,
        start_time: int = None,
        end_time: int = None,
        log_uri: Text = None,
        signature: Text = None
) -> JobMeta:
    return JobMeta(name=name, workflow_execution_id=workflow_execution_id, properties=properties, job_id=job_id,
                   start_time=start_time, end_time=end_time,
                   job_state=state, log_uri=log_uri, signature=signature)
