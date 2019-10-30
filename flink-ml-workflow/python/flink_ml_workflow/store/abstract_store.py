from abc import abstractmethod, ABCMeta
from python.flink_ml_workflow.vertex.model import Model, ModelVersion
from python.flink_ml_workflow.vertex.example import Example
from python.flink_ml_workflow.vertex.history import History
from python.flink_ml_workflow.vertex.project import Project


class AbstractStore:
    """
    Abstract class for Backend Storage.
    This class defines the API interface for front ends to connect with various types of backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Empty constructor for now. This is deliberately not marked as abstract, else every
        derived class would be forced to create one.
        """
        pass

    '''
        model api
    '''
    @abstractmethod
    def get_model_by_id(self, model_id):
        pass

    @abstractmethod
    def get_model_by_name(self, name):
        pass

    @abstractmethod
    def list_model(self, page_size, offset):
        pass

    @abstractmethod
    def list_model_version(self, model_id, page_size, offset):
        pass

    @abstractmethod
    def get_model_version_by_id(self, version_id):
        pass

    @abstractmethod
    def get_model_version_by_name(self, version_name):
        pass

    @abstractmethod
    def save_model(self, model: Model):
        pass

    @abstractmethod
    def save_model_version(self, model_version: ModelVersion):
        pass

    @abstractmethod
    def save_model_versions(self, version_list: list):
        pass

    '''
        example api
    '''

    @abstractmethod
    def get_example_by_id(self, experiment_id):
        pass

    @abstractmethod
    def get_example_by_name(self, experiment_name):
        pass

    @abstractmethod
    def list_example(self, page_size, offset):
        pass

    @abstractmethod
    def save_example(self, example: Example):
        pass

    '''
        project api
    '''

    @abstractmethod
    def get_project_by_id(self, project_id):
        pass

    @abstractmethod
    def get_project_by_name(self, project_name):
        pass

    @abstractmethod
    def save_project(self, project: Project):
        pass

    @abstractmethod
    def list_projects(self, page_size, offset):
        pass

    '''
        history api
    '''

    @abstractmethod
    def get_history_by_id(self, history_id):
        pass

    @abstractmethod
    def get_history_by_name(self, history_name):
        pass

    @abstractmethod
    def save_history(self, history: History):
        pass

    @abstractmethod
    def list_history(self, page_size, offset):
        pass
