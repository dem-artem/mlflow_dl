import os
import shutil
import subprocess
from os.path import join
from typing import Tuple, List, Set, Optional, Union

from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from tqdm import tqdm

from mlflow_dl.logger import logger


class MlflowModelVersionNotFoundError(Exception):
    pass


class MlflowHelper:
    _target_folder = None

    MODEL_FILENAME = "saved_model.pb"

    STAGE_PRODUCTION = "Production"
    STAGE_STAGING = "Staging"

    _client: MlflowClient = None

    def __init__(self, target_folder: str = None):
        self.target_folder = target_folder

    @property
    def target_folder(self) -> str:
        if self._target_folder is None:
            self._target_folder = os.environ.get("MLFLOWDL_TARGET_FOLDER_LOCAL", "models")
        return self._target_folder

    @target_folder.setter
    def target_folder(self, target_folder: str) -> None:
        self._target_folder = target_folder

    @property
    def client(self) -> MlflowClient:
        if self._client is None:
            self._client = MlflowClient()

        return self._client

    def get_latest_versions(self, name: str, stage: str) -> Union[List[ModelVersion], None]:
        model_info_list: list = self.client.get_latest_versions(  # noqa
            name, stages=[stage]
        )  # return type is list, not ModelVersion, mistake in return type of client
        models = None
        if len(model_info_list) > 0:
            models = model_info_list

        return models

    def download_artifacts(self, model_info: ModelVersion, remote_folder_name: str, dst_path: str) -> None:
        if "sftp://" in model_info.source:
            self.check_known_hosts(model_info.source)
        os.makedirs(dst_path, exist_ok=True)
        self.client.download_artifacts(model_info.run_id, remote_folder_name, dst_path)

    def download_model_with_metainfo(self, model_info: ModelVersion, dst_path: str) -> None:
        artifact_path = join(model_info.source.split("/")[-1])
        self.download_artifacts(model_info, artifact_path, dst_path)

    # def download_folder_by_version(self, remote_model: ModelVersion, remote_folder_name: str) -> Tuple[str, str]:
    #     dst_path = join(self.target_folder, remote_model.name, remote_model.version)
    #     if not os.path.isdir(dst_path):
    #         self.download_model_with_metainfo(remote_model, dst_path)
    #         tmp_model_path = join(dst_path, remote_folder_name)
    #         shutil.copytree(tmp_model_path, dst_path, dirs_exist_ok=True)
    #         shutil.rmtree(tmp_model_path)
    #         logger.info(f"Downloaded {remote_model.name} version {remote_model.version}")
    #     else:
    #         logger.warning(f"'{dst_path}' already exists, skipping")
    #
    #     return dst_path, remote_model.run_id

    def download_folder_by_model_version_sequence(
        self, model_versions: Tuple[Tuple[str, ModelVersion], ...], no_subfolder: bool = False
    ) -> Tuple[List[str], List[str]]:
        """Download a specific folder for model version

        :param model_versions: Tuple of tuples in ("folder_name", ModelVersion) format
        :return: Tuple with lists of target pathes for downloaded models and their run_ids
        """
        downloaded_models = []
        model_run_ids = []
        model_version: ModelVersion
        for folder_name, model_version in tqdm(model_versions):
            tmp_dst_path = self.download_folder_by_model_version(model_version, folder_name, no_subfolder=no_subfolder)
            downloaded_models.append(tmp_dst_path)
            model_run_ids.append(model_version.run_id)

        return downloaded_models, model_run_ids

    def download_folder_by_model_version(
        self, model_info: ModelVersion, remote_folder_name: str, no_subfolder: bool = False
    ) -> str:
        dst_path = join(self.target_folder, model_info.name, model_info.version)
        if not no_subfolder:
            dst_path = join(dst_path, remote_folder_name)
        if not os.path.isdir(dst_path):
            self.download_artifacts(model_info, remote_folder_name, dst_path)
            tmp_model_path = join(dst_path, remote_folder_name)
            shutil.copytree(tmp_model_path, dst_path, dirs_exist_ok=True)
            shutil.rmtree(tmp_model_path)
            logger.info(f"Downloaded {model_info.name} version {model_info.version}")
        else:
            logger.warning(f"'{dst_path}' already exists, skipping")

        return dst_path

    def download_models_by_version(
        self, models_to_download: Tuple[ModelVersion, ...], reformat_for_tf_serving: bool = False
    ) -> Tuple[List[str], List[str]]:
        remote_model_info: ModelVersion
        downloaded_models = []
        model_run_ids = []
        if len(models_to_download):
            for remote_model_info in tqdm(models_to_download):
                dst_path = join(self.target_folder, remote_model_info.name, remote_model_info.version)
                if not os.path.isdir(dst_path):
                    self.download_model_with_metainfo(remote_model_info, dst_path)
                    if reformat_for_tf_serving:
                        self.reformat_for_tf_serving_structure(remote_model_info, dst_path)
                    logger.info(f"Downloaded to '{self.target_folder}'")
                else:
                    logger.warning(f"'{dst_path}' already exists, skipping")
                downloaded_models.append(dst_path)
                model_run_ids.append(remote_model_info.run_id)

        else:
            logger.warning("Nothing to download")

        return downloaded_models, model_run_ids

    @staticmethod
    def reformat_for_tf_serving_structure(remote_model_info: ModelVersion, dst_path: str) -> None:
        artifact_model_name = remote_model_info.source.split("/")[-1]

        #  Mlflow can save models with a different structure, we need to cover all the cases
        tmp_model_path_option_1 = join(dst_path, artifact_model_name, "data", "model")
        tmp_model_path_option_2 = join(dst_path, artifact_model_name, "tfmodel")

        if os.path.isdir(tmp_model_path_option_1):
            shutil.copytree(tmp_model_path_option_1, dst_path, dirs_exist_ok=True)
            shutil.rmtree(tmp_model_path_option_1)
        elif os.path.isfile(tmp_model_path_option_1 + ".h5"):

            shutil.copyfile(
                f"{tmp_model_path_option_1}.h5",
                f"{dst_path}/model.h5",
            )
            shutil.rmtree(join(dst_path, artifact_model_name))
        elif os.path.isdir(tmp_model_path_option_2):
            shutil.copytree(tmp_model_path_option_2, dst_path, dirs_exist_ok=True)
            shutil.rmtree(tmp_model_path_option_2)
        else:
            shutil.rmtree(join(dst_path))
            raise FileNotFoundError

    def get_latest_models(self, remote_models: Set[str], is_staging: bool = False) -> List[ModelVersion]:
        models_to_download = []
        model_stage = self.STAGE_STAGING if is_staging else self.STAGE_PRODUCTION
        for model_name in remote_models:
            latest_version: ModelVersion = self.get_latest_versions(model_name, model_stage)
            if latest_version is not None:
                models_to_download.extend(latest_version)
            else:
                if model_stage == self.STAGE_STAGING:
                    # get production model if staging models doesn't exist
                    latest_version: ModelVersion = self.get_latest_versions(model_name, self.STAGE_PRODUCTION)
                if latest_version is not None:
                    models_to_download.extend(latest_version)
                else:
                    raise MlflowModelVersionNotFoundError(
                        f"Can't find a model '{model_name}' for a stage '{model_stage}'"
                    )

        return models_to_download

    def upload_folder_with_artifacts(self, run_id: str, local_folder_path: str, remote_folder_name: str) -> None:
        self.client.log_artifacts(run_id, local_folder_path, artifact_path=remote_folder_name)

    @staticmethod
    def check_known_hosts(connection_string: str) -> None:
        target_host = connection_string.split("/")[2].split("@")[-1].split(":")[0]
        file_path = os.path.expanduser("~/.ssh/known_hosts")
        is_target_host_known = False
        with open(file_path) as file:
            known_hosts = file.readlines()
        for line in known_hosts:
            hosts = line.split(" ")[0].split(",")
            if target_host in hosts:
                is_target_host_known = True
                break
        if not is_target_host_known:
            err, output = subprocess.getstatusoutput(f"ssh-keyscan {target_host} >> {file_path}")
            if err != 0:
                if output:
                    logger.error(output)
                raise RuntimeError("Can't add hostkey to known hosts.")
