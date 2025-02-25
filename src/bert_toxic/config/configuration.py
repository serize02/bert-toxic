from bert_toxic.constants import *
from bert_toxic.utils.common import read_yaml, create_directories
from bert_toxic.entity.entity_config import DataIngestionConfig, DataSplitConfig
from pathlib import Path

class ConfigurationManager:

    def __init__(self):

        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)   


    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file
        )

    def get_data_split_config(self) -> DataSplitConfig:

        config = self.config.data_split

        create_directories([config.root_dir, config.train_dir, config.test_dir])

        data_split_config = DataSplitConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(self.config.data_ingestion.local_data_file),
            train_dir=Path(config.train_dir),
            train_data_path=Path(config.train_data_path),
            test_dir=Path(config.test_dir),
            test_data_path=Path(config.test_data_path),
            params_train_size=self.params.TRAIN_SIZE
        )

        return data_split_config



