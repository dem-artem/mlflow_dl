from typing import Set, Tuple, List, Union

from mlflow_dl.mlflow_helper import MlflowHelper


class MlflowDl:
    _mlflow_helper = None
    _target_folder = None

    def __init__(self, target_folder: str = None):
        self._target_folder = target_folder

    @property
    def mlflow_helper(self) -> MlflowHelper:
        if self._mlflow_helper is None:
            self._mlflow_helper = MlflowHelper(target_folder=self._target_folder)

        return self._mlflow_helper

    def download_model_by_version(self, remote_model_name: str, version: str, no_subfolder: bool) -> Tuple[List[str], List[str]]:
        model_version = self.mlflow_helper.client.get_model_version(remote_model_name, version)

        return self.mlflow_helper.download_models_by_version((model_version,), no_subfolder=no_subfolder)

    def download_models_latest(
        self, remote_model_names: Set[str], no_subfolder: bool, is_staging: bool = False
    ) -> Tuple[List[str], List[str]]:
        latest_models = self.mlflow_helper.get_latest_models(remote_model_names, is_staging)

        return self.mlflow_helper.download_models_by_version(tuple(latest_models), no_subfolder=no_subfolder)

    def download_folder_by_model_version(
        self, remote_model_name: str, version: str, folder: str, no_subfolder: bool
    ) -> None:
        model_version = self.mlflow_helper.client.get_model_version(remote_model_name, version)
        self.mlflow_helper.download_folder_by_model_version(model_version, folder, no_subfolder=no_subfolder)

    def download_folder_by_models_versions(
        self,
        remote_model_names: Set[str],
        folder: Union[str, dict],
        no_subfolder: bool,
        is_staging: bool = False,
    ) -> Tuple[List[str], List[str]]:
        if isinstance(folder, str):
            folder_map = {model_name: folder for model_name in remote_model_names}
        else:
            if len(remote_model_names) == len(folder):
                folder_map = folder
            else:
                raise ValueError("Len for 'folder' and 'remote_model_names' must be the same!")

        latest_models = self.mlflow_helper.get_latest_models(remote_model_names, is_staging)
        model_folder_version_sequence = tuple(
            (folder_map[model_version.name], model_version) for model_version in latest_models
        )

        return self.mlflow_helper.download_folder_by_model_version_sequence(
            model_folder_version_sequence, no_subfolder=no_subfolder
        )
