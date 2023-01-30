import logging
import unittest
from copy import copy
from unittest.mock import MagicMock

from mlflow.entities.model_registry import ModelVersion

from mlflow_dl import MlflowDl
from mlflow_dl.logger import logger


class TestMlflowDl(unittest.TestCase):
    target_folder_name = "models"
    model_name = "fake_model_name"
    remote_model_folder_name = "model"
    version = "6"
    model_version = ModelVersion(
        creation_timestamp=1642711316194,
        current_stage="Production",
        description="",
        last_updated_timestamp=1644501303209,
        name=model_name,
        run_id="fake_run_id",
        run_link="",
        source=f"s3://_fake_bucket_name_/4d21c02c43b441a8a37f0a8ee2b1e2ed/artifacts/{remote_model_folder_name}",
        status="READY",
        status_message="",
        tags={},
        user_id="",
        version=version,
    )

    def setUp(self) -> None:
        self.mldl = MlflowDl()
        self.mldl._mlflow_helper = MagicMock()

    @classmethod
    def setUpClass(cls) -> None:
        logger.setLevel(logging.ERROR)

    def test_download_model_by_version(self) -> None:
        self.mldl.mlflow_helper.client.get_model_version = MagicMock(return_value=self.model_version)
        self.mldl.download_model_by_version(self.model_name, self.version)
        self.mldl.mlflow_helper.download_models_by_version.assert_called_with((self.model_version,))

    def test_download_models_latest(self) -> None:
        self.mldl.mlflow_helper.get_latest_models = MagicMock(return_value=[self.model_version])
        self.mldl.download_models_latest({self.model_name})
        self.mldl.mlflow_helper.download_models_by_version.assert_called_with((self.model_version,))

    def test_download_folder_by_model_version(self) -> None:
        self.mldl.mlflow_helper.client.get_model_version = MagicMock(return_value=self.model_version)
        self.mldl.download_folder_by_model_version(
            self.remote_model_folder_name, self.version, self.remote_model_folder_name, True
        )
        self.mldl.mlflow_helper.download_folder_by_model_version.assert_called_with(
            self.model_version, self.remote_model_folder_name, no_subfolder=True
        )

    def test_download_folder_by_models_versions_same_folder_name(self) -> None:
        models_names = ("fake_model_1", "fake_model_2")
        fake_model_version_1 = copy(self.model_version)
        fake_model_version_1._name = models_names[0]
        fake_model_version_2 = copy(self.model_version)
        fake_model_version_2._name = models_names[1]
        no_subfolder = True
        is_staging = True
        model_folder_version_sequence = (
            (self.remote_model_folder_name, fake_model_version_1),
            (self.remote_model_folder_name, fake_model_version_2),
        )
        self.mldl.mlflow_helper.get_latest_models = MagicMock(return_value=[fake_model_version_1, fake_model_version_2])
        self.mldl.download_folder_by_models_versions(
            set(models_names), self.remote_model_folder_name, no_subfolder=no_subfolder, is_staging=is_staging
        )
        self.mldl.mlflow_helper.download_folder_by_model_version_sequence.assert_called_with(
            model_folder_version_sequence, no_subfolder=no_subfolder
        )

    def test_download_folder_by_models_versions_diff_folder_names(self) -> None:
        models_names = ("fake_model_1", "fake_model_2")
        folder_names = ("optimized_folder", "not_optimized_folder")
        folder_names_dict = {item[0]: item[1] for item in zip(models_names, folder_names)}
        fake_model_version_1 = copy(self.model_version)
        fake_model_version_1._name = models_names[0]
        fake_model_version_2 = copy(self.model_version)
        fake_model_version_2._name = models_names[1]

        no_subfolder = True
        is_staging = True
        model_folder_version_sequence = (
            (folder_names[0], fake_model_version_1),
            (folder_names[1], fake_model_version_2),
        )
        self.mldl.mlflow_helper.get_latest_models = MagicMock(return_value=[fake_model_version_1, fake_model_version_2])
        self.mldl.download_folder_by_models_versions(
            set(models_names), folder_names_dict, no_subfolder=no_subfolder, is_staging=is_staging
        )
        self.mldl.mlflow_helper.download_folder_by_model_version_sequence.assert_called_with(
            model_folder_version_sequence, no_subfolder=no_subfolder
        )

    def test_download_folder_by_models_versions_exception(self) -> None:
        models_names = ("fake_model_1", "fake_model_2")
        folder_names_dict = {"fake_model_1": "fake_folder"}
        with self.assertRaises(ValueError) as context:
            self.mldl.download_folder_by_models_versions(set(models_names), folder_names_dict, no_subfolder=True)
        self.assertEqual("Len for 'folder' and 'remote_model_names' must be the same!", str(context.exception))
