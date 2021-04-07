import time
from typing import List

import ai_flow as af
from ai_flow import FunctionContext
from python_ai_flow import Executor


class SendEventExecutor(Executor):
    def __init__(self, key, value, event_type="UNDEFINED", num=1, pre_time=0, post_time=0):
        super().__init__()
        self.key = key
        self.value = value
        self.event_type = event_type
        self.num = num
        self.pre_time = pre_time
        self.post_time = post_time

    def execute(self, function_context: FunctionContext, input_list: List) -> List:
        for i in range(self.num):
            print("SendEventExecutor")
            time.sleep(self.pre_time)
            print('send event')
            af.send_event(self.key, self.value, self.event_type)
            time.sleep(self.post_time)
