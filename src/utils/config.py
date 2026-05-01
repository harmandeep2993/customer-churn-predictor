from pathlib import Path
import yaml
from dataclasses import dataclass

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass
class PathsConfig:
    raw_data: Path
    processed_data: Path
    features_data: Path
    models_dir: Path
    reports_dir: Path


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    model_name: str


@dataclass
class TrainingConfig:
    test_size: float
    random_state: int
    cv_folds: int


@dataclass
class Config:
    paths: PathsConfig
    mlflow: MLflowConfig
    training: TrainingConfig
    raw: dict  # full config accessible if needed


def load_config(path: Path = CONFIG_PATH) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    paths = PathsConfig(
        raw_data=Path(raw["paths"]["raw_data"]),
        processed_data=Path(raw["paths"]["processed_data"]),
        features_data=Path(raw["paths"]["features_data"]),
        models_dir=Path(raw["paths"]["models_dir"]),
        reports_dir=Path(raw["paths"]["reports_dir"]),
    )

    mlflow = MLflowConfig(
        tracking_uri=raw["mlflow"]["tracking_uri"],
        experiment_name=raw["mlflow"]["experiment_name"],
        model_name=raw["mlflow"]["model_name"],
    )

    training = TrainingConfig(
        test_size=raw["training"]["test_size"],
        random_state=raw["training"]["random_state"],
        cv_folds=raw["training"]["cv_folds"],
    )

    return Config(paths=paths, mlflow=mlflow, training=training, raw=raw)


config = load_config()