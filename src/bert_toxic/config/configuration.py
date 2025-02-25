from bert_toxic.constants import *
from bert_toxic.utils.common import read_yaml, create_directories
from bert_toxic.entity.entity_config import (DataIngestionConfig)
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


