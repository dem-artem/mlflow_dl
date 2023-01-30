import logging
import os
import unittest
from os.path import dirname, join
from unittest.mock import Mock, MagicMock, patch

from mlflow.entities.model_registry import ModelVersion

from mlflow_dl.logger import logger
from mlflow_dl.mlflow_helper import MlflowHelper, MlflowModelVersionNotFoundError


class TestMlflowHelper(unittest.TestCase):
    makedirs_patcher = None
    target_folder_name = "models"
    model_name = "fake_model_name"
    remote_model_folder_name = "model"
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
        version="6",
    )
    downloaded_path = f"{target_folder_name}/{model_version.name}/{model_version.version}"

    @classmethod
    def setUpClass(cls) -> None:
        logger.setLevel(logging.ERROR)
        cls.makedirs_patcher = patch("os.makedirs")
        cls.makedirs_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.makedirs_patcher.stop()

    def test_get_latest_models_production(self) -> None:
        remote_model_names = [self.model_name]
        mlflow_helper = MlflowHelper()
        mlflow_helper._client = Mock()
        mlflow_helper._client.get_latest_versions = MagicMock(return_value=[self.model_version])
        model_versions = mlflow_helper.get_latest_models(set(remote_model_names))
        mlflow_helper._client.get_latest_versions.assert_called_with(
            self.model_name, stages=[MlflowHelper.STAGE_PRODUCTION]
        )
        self.maxDiff = None
        self.assertSequenceEqual(model_versions, [self.model_version])

    def test_get_latest_models_stage(self) -> None:
        remote_model_names = [self.model_name]
        mlflow_helper = MlflowHelper()
        mlflow_helper._client = Mock()
        mlflow_helper._client.get_latest_versions = MagicMock(return_value=[self.model_version])
        model_versions = mlflow_helper.get_latest_models(set(remote_model_names), is_staging=True)
        mlflow_helper._client.get_latest_versions.assert_called_with(
            self.model_name, stages=[MlflowHelper.STAGE_STAGING]
        )
        self.maxDiff = None
        self.assertSequenceEqual(model_versions, [self.model_version])

    def _return_get_latest_versions_no_stage_prod_only(self, _: str, stages: str) -> list:  # noqa: U101
        out = []
        if stages == [MlflowHelper.STAGE_PRODUCTION]:
            out.append(self.model_version)

        return out

    def test_get_latest_models_prod_no_stage(self) -> None:
        remote_model_names = [self.model_name]
        mlflow_helper = MlflowHelper()
        mlflow_helper._client = Mock()
        mlflow_helper._client.get_latest_versions = MagicMock(
            side_effect=self._return_get_latest_versions_no_stage_prod_only
        )
        model_versions = mlflow_helper.get_latest_models(set(remote_model_names), is_staging=True)
        mlflow_helper._client.get_latest_versions.assert_called_with(
            self.model_name, stages=[MlflowHelper.STAGE_PRODUCTION]
        )
        self.assertSequenceEqual(model_versions, [self.model_version])

    def test_get_latest_models_empty_result(self) -> None:
        remote_model_names = [self.model_name]
        mlflow_helper = MlflowHelper()
        mlflow_helper._client = Mock()
        mlflow_helper._client.get_latest_versions = MagicMock(return_value=[])
        with self.assertRaises(MlflowModelVersionNotFoundError) as context:
            mlflow_helper.get_latest_models(set(remote_model_names))
            mlflow_helper._client.get_latest_versions.assert_called_with(
                self.model_name, stages=[MlflowHelper.STAGE_PRODUCTION]
            )
        self.assertTrue("Can't find a model 'fake_model_name' for a stage 'Production'" == str(context.exception))

    def test_download_model(self) -> None:
        dst_path = "fake_target_path"
        mlflow_helper = MlflowHelper()
        mlflow_helper._client = MagicMock()
        mlflow_helper.download_model_with_metainfo(self.model_version, dst_path)
        mlflow_helper._client.download_artifacts.assert_called_with(
            self.model_version.run_id, self.remote_model_folder_name, dst_path
        )

    @patch("subprocess.getstatusoutput")
    @patch("os.path.expanduser")
    def test_check_known_hosts_without_needed_host(self, known_hosts_file_mock: Mock, getstatusoutput: Mock) -> None:
        host = "0.0.0.0"
        connection_string = f"sftp://fake_user@{host}:22"
        known_host_file_path = dirname(__file__) + "/fixtures/known_hosts_without_needed"
        known_hosts_file_mock.return_value = known_host_file_path
        getstatusoutput.return_value = 0, ""
        MlflowHelper.check_known_hosts(connection_string)
        getstatusoutput.assert_called_with(f"ssh-keyscan {host} >> {known_host_file_path}")

    @patch("subprocess.getstatusoutput")
    @patch("os.path.expanduser")
    def test_check_known_hosts_exists(self, known_hosts_file_mock: Mock, getstatusoutput: Mock) -> None:
        host = "0.0.0.0"
        connection_string = f"sftp://fake_user@{host}:22"
        known_host_file_path = dirname(__file__) + "/fixtures/known_hosts"
        known_hosts_file_mock.return_value = known_host_file_path
        getstatusoutput.return_value = 0, ""
        MlflowHelper.check_known_hosts(connection_string)
        getstatusoutput.assert_not_called()

    @patch("subprocess.getstatusoutput")
    @patch("os.path.expanduser")
    def test_check_known_hosts_runtime_error(self, known_hosts_file_mock: Mock, getstatusoutput: Mock) -> None:
        host = "0.0.0.0"
        connection_string = f"sftp://fake_user@{host}:22"
        known_host_file_path = dirname(__file__) + "/fixtures/known_hosts_without_needed"
        known_hosts_file_mock.return_value = known_host_file_path
        getstatusoutput.return_value = 1, ""
        with self.assertRaises(RuntimeError) as context:
            MlflowHelper.check_known_hosts(connection_string)
        self.assertEqual("Can't add hostkey to known hosts.", str(context.exception))

    @patch("os.path.isdir")
    @patch("shutil.rmtree")
    @patch("shutil.copytree")
    def test_download_folder_by_model_version_sequence(
        self, copytree_mock: Mock, rmtree_mock: Mock, isdir_mock: Mock  # noqa U100
    ) -> None:
        isdir_mock.return_value = False
        fake_folder_name = "fake_folder"
        models_to_download = ((fake_folder_name, self.model_version),)
        os.environ["MLFLOWHELPER_TARGET_FOLDER_LOCAL"] = self.target_folder_name
        mlflow_helper = MlflowHelper()
        mlflow_helper._client = MagicMock()
        mlflow_helper.download_artifacts = MagicMock()
        mlflow_helper.download_folder_by_model_version_sequence(models_to_download)
        mlflow_helper.download_artifacts.assert_called_with(
            self.model_version,
            fake_folder_name,
            join(self.downloaded_path, fake_folder_name),
        )

    def test_download_models_by_version(self) -> None:
        mlflow_helper = MlflowHelper(target_folder=self.target_folder_name)
        mlflow_helper._client = MagicMock()
        mlflow_helper.download_model_with_metainfo = MagicMock()
        downloaded_path, run_id = mlflow_helper.download_models_by_version((self.model_version,))
        self.assertTrue(self.downloaded_path, downloaded_path)
        self.assertTrue(self.model_version.run_id, run_id)
