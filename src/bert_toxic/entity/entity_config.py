from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path


@dataclass
class DataSplitConfig:
    root_dir: Path
    data_path: Path
    train_dir: Path
    train_data_path: Path
    test_dir: Path
    test_data_path: Path
    params_train_size: float