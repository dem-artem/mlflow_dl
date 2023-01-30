import click

from dotenv import load_dotenv

load_dotenv(".env")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "-models",
    "-m",
    type=str,
    required=True,
    help="List of model names in the MLFlow Registry separated by comma",
)
@click.option("-o", "--output", "output", type=str, help="Folder name where place results")
@click.option("--no-subfolder", is_flag=True, help="Whether to place content in the root folder of model's version")
def download_production(models: str, output: str = None, no_subfolder: bool = None) -> None:
    """Download a list of models in status 'Production' (Tensorflow/Keras only tested)"""
    from mlflow_dl import MlflowDl

    models = tuple(str(model_name).strip() for model_name in models.split(","))
    MlflowDl(target_folder=output).download_models_latest(set(models), no_subfolder)


@cli.command()
@click.option(
    "-models",
    "-m",
    type=str,
    required=True,
    help="List of model names in the MLFlow Registry separated by comma",
)
@click.option("--output", "-o", type=str, help="Folder name where place results")
@click.option("--no-subfolder", is_flag=True, help="Whether to place content in the root folder of model's version")
def download_staging(models: str, output: str = None, no_subfolder: bool = None) -> None:
    """Download a list of models in status 'Staging' (Tensorflow/Keras only tested)"""
    from mlflow_dl import MlflowDl

    models = tuple(str(model_name).strip() for model_name in models.split(","))
    MlflowDl(target_folder=output).download_models_latest(set(models), no_subfolder, is_staging=True)


@cli.command()
@click.option("-model", "-m", type=str, required=True, help="Model name in the MLFlow Registry")
@click.option("-version", "-v", type=str, required=True, help="Model version MLFlow Registry")
@click.option("--output", "-o", type=str, help="Folder name where place results")
@click.option("--no-subfolder", is_flag=True, help="Whether to place content in the root folder of model's version")
def download_model_by_version(model: str, version: str, output: str = None, no_subfolder: bool = None) -> None:
    """Download a model by them registered model name and version (Tensorflow/Keras only tested)"""
    from mlflow_dl import MlflowDl

    MlflowDl(target_folder=output).download_model_by_version(model, version, no_subfolder)


@cli.command()
@click.option("-model", "-m", type=str, required=True, help="Model name in the MLFlow Registry")
@click.option("-version", "-v", type=str, required=True, help="Model version MLFlow Registry")
@click.option("-folder", "-f", type=str, required=True, help="Folder name in the root folder of model's version folder")
@click.option("--output", "-o", type=str, help="Folder name where place results")
@click.option("--no-subfolder", is_flag=True, help="Whether to place content in the root folder of model's version")
def download_folder_by_model_version(
        model: str, version: str, folder: str, output: str = None, no_subfolder: bool = None
) -> None:
    """Download a custom folder for a specific model's version"""
    from mlflow_dl import MlflowDl

    MlflowDl(target_folder=output).download_folder_by_model_version(model, version, folder, no_subfolder)
