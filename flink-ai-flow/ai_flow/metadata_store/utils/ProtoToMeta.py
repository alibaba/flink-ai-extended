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
from ai_flow.meta.artifact_meta import ArtifactMeta
from ai_flow.meta.example_meta import ExampleMeta, List, DataType, Schema, ExampleSupportType
from ai_flow.meta.job_meta import JobMeta, State
from ai_flow.meta.model_meta import ModelMeta, ModelVersionMeta
from ai_flow.meta.model_relation_meta import ModelRelationMeta, ModelVersionRelationMeta
from ai_flow.meta.project_meta import ProjectMeta
from ai_flow.meta.workflow_execution_meta import WorkflowExecutionMeta
from ai_flow.rest_endpoint.protobuf.message_pb2 import ExampleProto, ProjectProto, JobProto, \
    WorkflowExecutionProto, StateProto, ModelRelationProto, ModelVersionRelationProto, ModelProto, ModelVersionProto, \
    ArtifactProto, ModelVersionStage, DataTypeProto, ExampleSupportTypeProto, ModelType


class ProtoToMeta:

    @staticmethod
    def proto_to_example_meta(example_proto: ExampleProto) -> ExampleMeta:
        support_type = ExampleSupportType(ExampleSupportTypeProto.Name(example_proto.support_type))
        properties = example_proto.properties
        if properties == {}:
            properties = None
        name_list = example_proto.schema.name_list
        if not name_list:
            name_list = None
        type_list = example_proto.schema.type_list
        if not type_list:
            data_type_list = None
        else:
            data_type_list = []
            for c in type_list:
                data_type_list.append(DataType(DataTypeProto.Name(c)))
        data_type = example_proto.data_type.value if example_proto.HasField('data_type') else None
        data_format = example_proto.data_format.value if example_proto.HasField('data_format') else None
        description = example_proto.description.value if example_proto.HasField('description') else None
        batch_uri = example_proto.batch_uri.value if example_proto.HasField('batch_uri') else None
        stream_uri = example_proto.stream_uri.value if example_proto.HasField('stream_uri') else None
        create_time = example_proto.create_time.value if example_proto.HasField('create_time') else None
        update_time = example_proto.update_time.value if example_proto.HasField('update_time') else None
        catalog_name = example_proto.catalog_name.value if example_proto.HasField('catalog_name') else None
        catalog_type = example_proto.catalog_type.value if example_proto.HasField('catalog_type') else None
        catalog_database = example_proto.catalog_database.value if example_proto.HasField('catalog_database') else None
        catalog_version = example_proto.catalog_version.value if example_proto.HasField('catalog_version') else None
        catalog_connection_uri = example_proto.catalog_connection_uri.value \
            if example_proto.HasField('catalog_connection_uri') else None
        catalog_table = example_proto.catalog_table.value if example_proto.HasField('catalog_table') else None
        schema = Schema(name_list=name_list, type_list=data_type_list)
        return ExampleMeta(uuid=example_proto.uuid,
                           name=example_proto.name,
                           support_type=support_type,
                           data_type=data_type,
                           data_format=data_format,
                           description=description,
                           batch_uri=batch_uri,
                           stream_uri=stream_uri,
                           create_time=create_time,
                           update_time=update_time,
                           properties=properties,
                           schema=schema,
                           catalog_name=catalog_name,
                           catalog_type=catalog_type,
                           catalog_database=catalog_database,
                           catalog_version=catalog_version,
                           catalog_connection_uri=catalog_connection_uri,
                           catalog_table=catalog_table)

    @staticmethod
    def proto_to_example_meta_list(example_proto_list: List[ExampleProto]) -> List[ExampleMeta]:
        list_example_meta = []
        for example_proto in example_proto_list:
            list_example_meta.append(ProtoToMeta.proto_to_example_meta(example_proto))
        return list_example_meta

    @staticmethod
    def proto_to_project_meta(project_proto: ProjectProto) -> ProjectMeta:
        properties = project_proto.properties
        if properties == {}:
            properties = None
        return ProjectMeta(
            uuid=project_proto.uuid,
            name=project_proto.name,
            properties=properties,
            uri=project_proto.uri.value if project_proto.HasField('uri') else None,
            user=project_proto.user.value if project_proto.HasField('user') else None,
            password=project_proto.password.value if project_proto.HasField('password') else None,
            project_type=project_proto.project_type.value if project_proto.HasField(
                'project_type') else None)

    @staticmethod
    def proto_to_project_meta_list(projects: List[ProjectProto]) -> List[ProjectMeta]:
        project_meta_list = []
        for project in projects:
            project_meta_list.append(ProtoToMeta.proto_to_project_meta(project))
        return project_meta_list

    @staticmethod
    def proto_to_artifact_meta(artifact_proto: ArtifactProto) -> ArtifactMeta:
        properties = artifact_proto.properties
        if properties == {}:
            properties = None
        return ArtifactMeta(
            uuid=artifact_proto.uuid,
            name=artifact_proto.name,
            properties=properties,
            data_format=artifact_proto.data_format.value if artifact_proto.HasField('data_format') else None,
            description=artifact_proto.description.value if artifact_proto.HasField('description') else None,
            batch_uri=artifact_proto.batch_uri.value if artifact_proto.HasField('batch_uri') else None,
            stream_uri=artifact_proto.stream_uri.value if artifact_proto.HasField('stream_uri') else None,
            create_time=artifact_proto.create_time.value if artifact_proto.HasField('create_time') else None,
            update_time=artifact_proto.update_time.value if artifact_proto.HasField('update_time') else None)

    @staticmethod
    def proto_to_artifact_meta_list(artifacts: List[ArtifactProto]) -> List[ArtifactMeta]:
        artifact_meta_list = []
        for artifact in artifacts:
            artifact_meta_list.append(ProtoToMeta.proto_to_artifact_meta(artifact))
        return artifact_meta_list

    @staticmethod
    def proto_to_job_meta(job_proto: JobProto) -> JobMeta:
        job_state = State(StateProto.Name(job_proto.job_state))
        properties = job_proto.properties
        if properties == {}:
            properties = None
        return JobMeta(
            uuid=job_proto.uuid,
            name=job_proto.name,
            job_state=job_state,
            properties=properties,
            workflow_execution_id=job_proto.workflow_execution_id.value if job_proto.HasField(
                'workflow_execution_id') else None,
            job_id=job_proto.job_id.value if job_proto.HasField('job_id') else None,
            start_time=job_proto.start_time.value if job_proto.HasField(
                'start_time') else None,
            end_time=job_proto.end_time.value if job_proto.HasField(
                'end_time') else None,
            log_uri=job_proto.log_uri.value if job_proto.HasField(
                'log_uri') else None,
            signature=job_proto.signature.value if job_proto.HasField(
                'signature') else None)

    @staticmethod
    def proto_to_job_meta_list(jobs: List[JobProto]) -> List[JobMeta]:
        job_meta_list = []
        for job in jobs:
            job_meta_list.append(ProtoToMeta.proto_to_job_meta(job))
        return job_meta_list

    @staticmethod
    def proto_to_execution_meta(workflow_execution_proto: WorkflowExecutionProto) -> WorkflowExecutionMeta:
        execution_state = State(StateProto.Name(workflow_execution_proto.execution_state))
        properties = workflow_execution_proto.properties
        if properties == {}:
            properties = None
        return WorkflowExecutionMeta(
            uuid=workflow_execution_proto.uuid,
            name=workflow_execution_proto.name,
            execution_state=execution_state,
            properties=properties,
            project_id=workflow_execution_proto.project_id.value if workflow_execution_proto.HasField(
                'project_id') else None,
            start_time=workflow_execution_proto.start_time.value if workflow_execution_proto.HasField(
                'start_time') else None,
            end_time=workflow_execution_proto.end_time.value if workflow_execution_proto.HasField(
                'end_time') else None,
            log_uri=workflow_execution_proto.log_uri.value if workflow_execution_proto.HasField(
                'log_uri') else None,
            workflow_json=workflow_execution_proto.workflow_json.value if workflow_execution_proto.HasField(
                'workflow_json') else None,
            signature=workflow_execution_proto.signature.value if workflow_execution_proto.HasField(
                'signature') else None)

    @staticmethod
    def proto_to_execution_meta_list(executions: List[WorkflowExecutionProto]) -> List[WorkflowExecutionMeta]:
        execution_meta_list = []
        for execution in executions:
            execution_meta_list.append(ProtoToMeta.proto_to_execution_meta(execution))
        return execution_meta_list

    @staticmethod
    def proto_to_state(state):
        if state == StateProto.INIT:
            return State.INIT
        elif state == StateProto.STARTING:
            return State.STARTING
        elif state == StateProto.RUNNING:
            return State.RUNNING
        elif state == StateProto.FINISHED:
            return State.FINISHED
        elif state == StateProto.FAILED:
            return State.FAILED
        elif state == StateProto.KILLING:
            return State.KILLING
        elif state == StateProto.KILLED:
            return State.KILLED

    @staticmethod
    def proto_to_model_relation_meta(model_relation_proto: ModelRelationProto) -> ModelRelationMeta:
        return ModelRelationMeta(uuid=model_relation_proto.uuid,
                                 name=model_relation_proto.name,
                                 project_id=model_relation_proto.project_id.value
                                 if model_relation_proto.HasField('project_id') else None)

    @staticmethod
    def proto_to_model_relation_meta_list(model_proto_relation_list: List[ModelRelationProto]) -> List[
        ModelRelationMeta]:
        model_relation_meta_list = []
        for model_proto in model_proto_relation_list:
            model_relation_meta_list.append(ProtoToMeta.proto_to_model_relation_meta(model_proto))
        return model_relation_meta_list

    @staticmethod
    def proto_to_model_meta(model_proto: ModelProto) -> ModelMeta:
        return ModelMeta(uuid=model_proto.uuid, name=model_proto.name,
                         model_type=ModelType.Name(model_proto.model_type),
                         model_desc=model_proto.model_desc.value if model_proto.HasField(
                             'model_desc') else None,
                         project_id=model_proto.project_id.value if model_proto.HasField('project_id') else None)

    @staticmethod
    def proto_to_model_meta_list(model_proto_list: List[ModelProto]) -> List[ModelMeta]:
        model_meta_list = []
        for model_proto in model_proto_list:
            model_meta_list.append(ProtoToMeta.proto_to_model_meta(model_proto))
        return model_meta_list

    @staticmethod
    def proto_to_model_version_relation_meta(
            model_version_proto: ModelVersionRelationProto) -> ModelVersionRelationMeta:
        return ModelVersionRelationMeta(
            version=model_version_proto.version.value if model_version_proto.HasField('version') else None,
            model_id=model_version_proto.model_id.value if model_version_proto.HasField('model_id') else None,
            workflow_execution_id=model_version_proto.workflow_execution_id.value
            if model_version_proto.HasField('workflow_execution_id') else None)

    @staticmethod
    def proto_to_model_version_relation_meta_list(model_version_proto_list: List[ModelVersionRelationProto]) -> List[
        ModelVersionRelationMeta]:
        model_version_relation_meta_list = []
        for model_version_proto in model_version_proto_list:
            model_version_relation_meta_list.append(
                ProtoToMeta.proto_to_model_version_relation_meta(model_version_proto))
        return model_version_relation_meta_list

    @staticmethod
    def proto_to_model_version_meta(
            model_version_proto: ModelVersionProto) -> ModelVersionMeta:
        return ModelVersionMeta(
            version=model_version_proto.version.value if model_version_proto.HasField(
                'version') else None,
            model_id=model_version_proto.model_id.value if model_version_proto.HasField('model_id') else None,
            model_path=model_version_proto.model_path.value if model_version_proto.HasField(
                'model_path') else None,
            model_metric=model_version_proto.model_metric.value if model_version_proto.HasField(
                'model_metric') else None,
            model_flavor=model_version_proto.model_flavor.value if model_version_proto.HasField(
                'model_flavor') else None,
            workflow_execution_id=model_version_proto.workflow_execution_id.value if model_version_proto.HasField(
                'workflow_execution_id') else None,
            version_desc=model_version_proto.version_desc.value if model_version_proto.HasField(
                'version_desc') else None,
            current_stage=ModelVersionStage.Name(model_version_proto.current_stage))

    @staticmethod
    def proto_to_model_version_meta_list(model_version_proto_list: List[ModelVersionProto]) -> List[ModelVersionMeta]:
        model_version_meta_list = []
        for model_version_proto in model_version_proto_list:
            model_version_meta_list.append(
                ProtoToMeta.proto_to_model_version_meta(model_version_proto))
        return model_version_meta_list
