from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path


@dataclass(frozen=True)
class DataSplitConfig:
    root_dir: Path
    data_path: Path
    train_dir: Path
    train_data_path: Path
    test_dir: Path
    test_data_path: Path
    params_train_size: float


@dataclass(frozen=True)
class SetupModelConfig:
    root_dir: Path
    model_path: Path
    params_classes: int
    params_dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    model_path: Path
    trained_model_path: Path
    training_data_path: Path
    params_max_len: int
    params_train_batch_size: int
    params_epochs: int
    params_learning_rate: float
    params_train_num_workers: int
    params_train_shuffle: bool


@dataclass(frozen=True)
class EvaluationConfig:
    model_path: Path
    testing_data_path: Path
    all_params: dict
    params_max_len: int
    params_valid_batch_size: int
    params_epochs: int
    params_valid_num_workers: int
    params_valid_shuffle: bool