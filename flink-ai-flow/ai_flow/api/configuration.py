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
from typing import Text, Optional
import os
from ai_flow.project.project_description import get_project_description_from, ProjectDesc

from ai_flow.project.project_config import ProjectConfig, _default_project_config
from ai_flow.rest_endpoint.service.client.aiflow_client import AIFlowClient

_default_project_config_set_flag = False


def ensure_project_registered():
    """ Ensure the project configured in project.yaml has been registered. """

    client = AIFlowClient(server_uri=_default_project_config.get_master_uri())
    project_meta = client.get_project_by_name(_default_project_config.get_project_name())
    pp = {}
    for k, v in _default_project_config.items():
        pp[k] = str(v)
    if project_meta is None:
        project_meta = client.register_project(name=_default_project_config.get_project_name(),
                                               properties=pp)
    else:
        project_meta = client.update_project(project_name=_default_project_config.get_project_name(), properties=pp)

    _default_project_config.set_project_uuid(str(project_meta.uuid))


def project_config() -> Optional[ProjectConfig]:
    """
    :return: project configuration
    """
    ensure_project_registered()
    return _default_project_config


def set_project_config_file(config_file: Text):
    """
    Set project config based on project.yaml in client.

    :param config_file: Path to project.yaml
    :return: None
    """
    global _default_project_config_set_flag
    if _default_project_config_set_flag:
        raise Exception("project configuration cannot be set repeatedly!")
    else:
        _default_project_config.load_from_file(config_file)
        _default_project_config_set_flag = True


def set_default_project_config(config: ProjectConfig):
    global _default_project_config_set_flag
    if _default_project_config_set_flag:
        raise Exception("project configuration cannot be set repeatedly!")
    else:
        _default_project_config.clear()
        _default_project_config.update(config)
        _default_project_config_set_flag = True


def get_default_project_config():
    return _default_project_config


def set_project_master_uri(master_uri: Text):
    ip_port = master_uri.split(':')
    _default_project_config.set_master_ip(ip_port[0])
    _default_project_config.set_master_port(ip_port[1])


def unset_project_config():
    global _default_project_config_set_flag, _default_project_config
    _default_project_config = ProjectConfig()
    _default_project_config_set_flag = False


_default_project_desc: ProjectDesc = None


def project_description()->ProjectDesc:
    return _default_project_desc


def set_project_path(path):
    global _default_project_desc
    set_project_config_file(os.path.join(path, 'project.yaml'))
    _default_project_desc = get_project_description_from(path)
    _default_project_desc.project_config = _default_project_config
