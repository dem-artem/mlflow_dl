# MLFlow model downloader
Allows to download models from [MLFlow](https://mlflow.org/) Model Registry using model's status (Staging or Production) or model's Version.

**NOTES:** 
- supports MLflow's version <1.20.0
- tested only with Tensorflow/Keras models
- tested on S3 and SFTP storages

## Installation

1. Installation for production from cloned repository

```shell
cd mlflow_dl
pip install .
```

2. Installation for development from cloned repository

```shell
cd mlflow_dl
pip install -e ".[dev]"
```

## Environment Variables Configuration

In the most of the cases you need to configure a proper credentials to mlflow and related storage.
Built in variables:

| ENV Name                     | Description                       
|------------------------------|-----------------------------------|
| MLFLOWDL_TARGET_FOLDER_LOCAL | The folder where downloaded results placed 

Also, you may need to configure variable for access to MLFlow and AWS S3 bucket. Some of them:

| ENV Name                         | Description                                                                                                                  |
|----------------------------------|---------------------------------------|
| AWS_ACCESS_KEY_ID                |
| AWS_SECRET_ACCESS_KEY            |
| AWS_DEFAULT_REGION               |
| AWS_BUCKET_NAME                  |
| MLFLOW_TRACKING_USERNAME         | Username for HTTP Base Auth for MLFlow (if used)
| MLFLOW_TRACKING_PASSWORD         | Password for HTTP Base Auth for MLFlow (if used)
| MLFLOW_TRACKING_URI              | URL to MLFlow UI (required)

## Usage from Command Line

- Get list of available commands:

```shell
python -m mlflow_dl
```

or

```shell
mlflow_dl
```

- Download models in "Production" stage:

```shell
mlflow_dl download-models-production-latest -m model_name_1,model_name_2
```

- Download models in "Staging" stage:

```shell
mlflow_dl download-models-staging-latest -m model_name_1,model_name_2
```

- Download model by name and version number:

```shell
mlflow_dl download-model-by-version -m model_name -v 1
```

- Download specific folder from the root of model's experiment:

```shell
mlflow_dl download-folder-by-model-version -m model_name -v 1 -f custom_folder
```

- Download specific folder for [Tensorflow Serving compatible structure](https://www.tensorflow.org/tfx/serving/serving_basic#load_exported_model_with_standard_tensorflow_modelserver):

```shell
mlflow_dl download-folder-by-model-version -m model_name -v 1 -f custom_folder --no-subfolder
```

## Usage from Code

```python
from mlflow_dl import MlflowDl

mlflowdl = MlflowDl(target_folder="tmp_folder")
mlflowdl.download_model_by_version("model_name", "1")
```

## Development

### Testing

```shell
python -m unittest
```

### Coverage

```shell
python -m coverage run -m unittest
python -m coverage html -i
```

### Code Style

```shell
python -m flake8
```

### Versioning

Commit your changes and run with a proper label (major | minor | patch):

```shell
bumpversion patch
git push origin
git push origin --tags
```

### Build package

```shell
rm -rf ./dist
python -m build
```

### Upload to custom PyPI
In case of private repo you need to make sure you configured credentials in `~/.pypirc`, than run:

```shell
twine upload --repository {repo_name} dist/* 
```
