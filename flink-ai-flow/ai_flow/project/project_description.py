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
import os
from typing import List, Text
from ai_flow.project.project_config import ProjectConfig, _default_project_config
from ai_flow.common.json_utils import Jsonable
from pathlib import Path
import logging


class ProjectDesc(Jsonable):
    """
    ProjectDesc( project descriptor) maintains the directory structure and information of an ai flow project.
    A common ai flow project would follow the directory structure as bellow.

    SimpleProject
    ├─ project.yaml
    ├─ jar_dependencies
    ├─ resources
    ├─ temp
    ├─ logs
    └─ python_codes
       ├─ __init__.py
       ├─ my_ai_flow.py
       └─ requirements.txt
    """
    def __init__(self) -> None:
        super().__init__()
        self.project_name: Text = None
        self.project_path = '/tmp'
        self.python_dependencies: List[Text] = []
        self.jar_dependencies: List[Text] = []
        self.resources: List[Text] = []
        self.project_temp_path: Text = 'temp'
        self.log_path: Text = "logs"
        self.project_config: ProjectConfig = None
        self.python_paths: List[Text] = []

    def get_absolute_temp_path(self)->Text:
        return "{}/{}".format(self.project_path, self.project_temp_path)

    def get_absolute_log_path(self)->Text:
        return "{}/{}".format(self.project_path, self.log_path)


def get_project_description_from(project_path: Text) -> ProjectDesc:
    """
    Load a project descriptor for a given project path.
    :param project_path: the path of a ai flow project.
    :return: a ProjectDesc object that contains the structure information of this project.
    """
    project_spec = ProjectDesc()
    project_path = os.path.abspath(project_path)
    project_spec.project_path = project_path
    project_path_obj = Path(project_path)
    project_spec.jar_dependencies = get_file_paths_from(str(project_path_obj / 'jar_dependencies'))
    project_spec.python_dependencies = get_file_paths_from(str(project_path_obj / 'python_codes'))
    project_spec.resources = get_file_paths_from(str(project_path_obj / 'resources'))
    if not os.path.exists(project_spec.get_absolute_temp_path()):
        os.makedirs(project_spec.get_absolute_temp_path())
    project_spec.project_config = ProjectConfig()
    project_spec.project_config.load_from_file(os.path.join(project_path, 'project.yaml'))
    # adapter to old scheduler
    if _default_project_config.get_project_uuid() is not None:
        project_spec.project_config.set_project_uuid(_default_project_config.get_project_uuid())
    if 'entry_module_path' in _default_project_config:
        project_spec.project_config['entry_module_path'] = _default_project_config['entry_module_path']
    project_spec.project_name = project_spec.project_config.get_project_name()
    return project_spec


def get_file_paths_from(dir: Text) -> List[Text]:
    """
    list all file paths inside a directory.

    :param dir: a directory path that need to list.
    :return: a string list of file paths.
    """
    if not os.path.exists(dir):
        logging.info('{} does not exist.'.format(dir))
        return None
    file_paths = ["{}/{}".format(dir, x) for x in os.listdir(dir)]
    return file_paths


