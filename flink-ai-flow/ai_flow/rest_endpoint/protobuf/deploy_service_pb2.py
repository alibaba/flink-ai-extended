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

# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: deploy_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import service as _service
from google.protobuf import service_reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='deploy_service.proto',
  package='ai_flow',
  syntax='proto3',
  serialized_pb=_b('\n\x14\x64\x65ploy_service.proto\x12\x07\x61i_flow\x1a\x1cgoogle/api/annotations.proto\"4\n\x0fWorkflowRequest\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x15\n\rworkflow_json\x18\x02 \x01(\t\"I\n\x10ScheduleResponse\x12\x13\n\x0breturn_code\x18\x01 \x01(\x03\x12\x12\n\nreturn_msg\x18\x02 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\t\"!\n\x13MasterConfigRequest\x12\n\n\x02id\x18\x01 \x01(\x03\"\xa9\x01\n\x14MasterConfigResponse\x12\x13\n\x0breturn_code\x18\x01 \x01(\x03\x12\x12\n\nreturn_msg\x18\x02 \x01(\t\x12\x39\n\x06\x63onfig\x18\x03 \x03(\x0b\x32).ai_flow.MasterConfigResponse.ConfigEntry\x1a-\n\x0b\x43onfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x32\x83\x05\n\rDeployService\x12x\n\x15startScheduleWorkflow\x12\x18.ai_flow.WorkflowRequest\x1a\x19.ai_flow.ScheduleResponse\"*\x82\xd3\xe4\x93\x02$\"\x1f/aiflow/deployer/workflow/start:\x01*\x12v\n\x14stopScheduleWorkflow\x12\x18.ai_flow.WorkflowRequest\x1a\x19.ai_flow.ScheduleResponse\")\x82\xd3\xe4\x93\x02#\"\x1e/aiflow/deployer/workflow/stop:\x01*\x12~\n\x1agetWorkflowExecutionResult\x12\x18.ai_flow.WorkflowRequest\x1a\x19.ai_flow.ScheduleResponse\"+\x82\xd3\xe4\x93\x02%\" /aiflow/deployer/workflow/result:\x01*\x12{\n\x18isWorkflowExecutionAlive\x12\x18.ai_flow.WorkflowRequest\x1a\x19.ai_flow.ScheduleResponse\"*\x82\xd3\xe4\x93\x02$\"\x1f/aiflow/deployer/workflow/alive:\x01*\x12\x82\x01\n\x0fgetMasterConfig\x12\x1c.ai_flow.MasterConfigRequest\x1a\x1d.ai_flow.MasterConfigResponse\"2\x82\xd3\xe4\x93\x02,\"\'/aiflow/deployer/workflow/master_config:\x01*B\"\n\x10\x63om.aiflow.protoZ\x08/ai_flow\x88\x01\x01\x90\x01\x01\x62\x06proto3')
  ,
  dependencies=[google_dot_api_dot_annotations__pb2.DESCRIPTOR,])




_WORKFLOWREQUEST = _descriptor.Descriptor(
  name='WorkflowRequest',
  full_name='ai_flow.WorkflowRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='ai_flow.WorkflowRequest.id', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='workflow_json', full_name='ai_flow.WorkflowRequest.workflow_json', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=63,
  serialized_end=115,
)


_SCHEDULERESPONSE = _descriptor.Descriptor(
  name='ScheduleResponse',
  full_name='ai_flow.ScheduleResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='return_code', full_name='ai_flow.ScheduleResponse.return_code', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='return_msg', full_name='ai_flow.ScheduleResponse.return_msg', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='ai_flow.ScheduleResponse.data', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=117,
  serialized_end=190,
)


_MASTERCONFIGREQUEST = _descriptor.Descriptor(
  name='MasterConfigRequest',
  full_name='ai_flow.MasterConfigRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='ai_flow.MasterConfigRequest.id', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=192,
  serialized_end=225,
)


_MASTERCONFIGRESPONSE_CONFIGENTRY = _descriptor.Descriptor(
  name='ConfigEntry',
  full_name='ai_flow.MasterConfigResponse.ConfigEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='ai_flow.MasterConfigResponse.ConfigEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='ai_flow.MasterConfigResponse.ConfigEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=352,
  serialized_end=397,
)

_MASTERCONFIGRESPONSE = _descriptor.Descriptor(
  name='MasterConfigResponse',
  full_name='ai_flow.MasterConfigResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='return_code', full_name='ai_flow.MasterConfigResponse.return_code', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='return_msg', full_name='ai_flow.MasterConfigResponse.return_msg', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='config', full_name='ai_flow.MasterConfigResponse.config', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_MASTERCONFIGRESPONSE_CONFIGENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=228,
  serialized_end=397,
)

_MASTERCONFIGRESPONSE_CONFIGENTRY.containing_type = _MASTERCONFIGRESPONSE
_MASTERCONFIGRESPONSE.fields_by_name['config'].message_type = _MASTERCONFIGRESPONSE_CONFIGENTRY
DESCRIPTOR.message_types_by_name['WorkflowRequest'] = _WORKFLOWREQUEST
DESCRIPTOR.message_types_by_name['ScheduleResponse'] = _SCHEDULERESPONSE
DESCRIPTOR.message_types_by_name['MasterConfigRequest'] = _MASTERCONFIGREQUEST
DESCRIPTOR.message_types_by_name['MasterConfigResponse'] = _MASTERCONFIGRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

WorkflowRequest = _reflection.GeneratedProtocolMessageType('WorkflowRequest', (_message.Message,), dict(
  DESCRIPTOR = _WORKFLOWREQUEST,
  __module__ = 'deploy_service_pb2'
  # @@protoc_insertion_point(class_scope:ai_flow.WorkflowRequest)
  ))
_sym_db.RegisterMessage(WorkflowRequest)

ScheduleResponse = _reflection.GeneratedProtocolMessageType('ScheduleResponse', (_message.Message,), dict(
  DESCRIPTOR = _SCHEDULERESPONSE,
  __module__ = 'deploy_service_pb2'
  # @@protoc_insertion_point(class_scope:ai_flow.ScheduleResponse)
  ))
_sym_db.RegisterMessage(ScheduleResponse)

MasterConfigRequest = _reflection.GeneratedProtocolMessageType('MasterConfigRequest', (_message.Message,), dict(
  DESCRIPTOR = _MASTERCONFIGREQUEST,
  __module__ = 'deploy_service_pb2'
  # @@protoc_insertion_point(class_scope:ai_flow.MasterConfigRequest)
  ))
_sym_db.RegisterMessage(MasterConfigRequest)

MasterConfigResponse = _reflection.GeneratedProtocolMessageType('MasterConfigResponse', (_message.Message,), dict(

  ConfigEntry = _reflection.GeneratedProtocolMessageType('ConfigEntry', (_message.Message,), dict(
    DESCRIPTOR = _MASTERCONFIGRESPONSE_CONFIGENTRY,
    __module__ = 'deploy_service_pb2'
    # @@protoc_insertion_point(class_scope:ai_flow.MasterConfigResponse.ConfigEntry)
    ))
  ,
  DESCRIPTOR = _MASTERCONFIGRESPONSE,
  __module__ = 'deploy_service_pb2'
  # @@protoc_insertion_point(class_scope:ai_flow.MasterConfigResponse)
  ))
_sym_db.RegisterMessage(MasterConfigResponse)
_sym_db.RegisterMessage(MasterConfigResponse.ConfigEntry)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\020com.aiflow.protoZ\010/ai_flow\210\001\001\220\001\001'))
_MASTERCONFIGRESPONSE_CONFIGENTRY.has_options = True
_MASTERCONFIGRESPONSE_CONFIGENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))

_DEPLOYSERVICE = _descriptor.ServiceDescriptor(
  name='DeployService',
  full_name='ai_flow.DeployService',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=400,
  serialized_end=1043,
  methods=[
  _descriptor.MethodDescriptor(
    name='startScheduleWorkflow',
    full_name='ai_flow.DeployService.startScheduleWorkflow',
    index=0,
    containing_service=None,
    input_type=_WORKFLOWREQUEST,
    output_type=_SCHEDULERESPONSE,
    options=_descriptor._ParseOptions(descriptor_pb2.MethodOptions(), _b('\202\323\344\223\002$\"\037/aiflow/deployer/workflow/start:\001*')),
  ),
  _descriptor.MethodDescriptor(
    name='stopScheduleWorkflow',
    full_name='ai_flow.DeployService.stopScheduleWorkflow',
    index=1,
    containing_service=None,
    input_type=_WORKFLOWREQUEST,
    output_type=_SCHEDULERESPONSE,
    options=_descriptor._ParseOptions(descriptor_pb2.MethodOptions(), _b('\202\323\344\223\002#\"\036/aiflow/deployer/workflow/stop:\001*')),
  ),
  _descriptor.MethodDescriptor(
    name='getWorkflowExecutionResult',
    full_name='ai_flow.DeployService.getWorkflowExecutionResult',
    index=2,
    containing_service=None,
    input_type=_WORKFLOWREQUEST,
    output_type=_SCHEDULERESPONSE,
    options=_descriptor._ParseOptions(descriptor_pb2.MethodOptions(), _b('\202\323\344\223\002%\" /aiflow/deployer/workflow/result:\001*')),
  ),
  _descriptor.MethodDescriptor(
    name='isWorkflowExecutionAlive',
    full_name='ai_flow.DeployService.isWorkflowExecutionAlive',
    index=3,
    containing_service=None,
    input_type=_WORKFLOWREQUEST,
    output_type=_SCHEDULERESPONSE,
    options=_descriptor._ParseOptions(descriptor_pb2.MethodOptions(), _b('\202\323\344\223\002$\"\037/aiflow/deployer/workflow/alive:\001*')),
  ),
  _descriptor.MethodDescriptor(
    name='getMasterConfig',
    full_name='ai_flow.DeployService.getMasterConfig',
    index=4,
    containing_service=None,
    input_type=_MASTERCONFIGREQUEST,
    output_type=_MASTERCONFIGRESPONSE,
    options=_descriptor._ParseOptions(descriptor_pb2.MethodOptions(), _b('\202\323\344\223\002,\"\'/aiflow/deployer/workflow/master_config:\001*')),
  ),
])
_sym_db.RegisterServiceDescriptor(_DEPLOYSERVICE)

DESCRIPTOR.services_by_name['DeployService'] = _DEPLOYSERVICE

DeployService = service_reflection.GeneratedServiceType('DeployService', (_service.Service,), dict(
  DESCRIPTOR = _DEPLOYSERVICE,
  __module__ = 'deploy_service_pb2'
  ))

DeployService_Stub = service_reflection.GeneratedServiceStubType('DeployService_Stub', (DeployService,), dict(
  DESCRIPTOR = _DEPLOYSERVICE,
  __module__ = 'deploy_service_pb2'
  ))


# @@protoc_insertion_point(module_scope)
