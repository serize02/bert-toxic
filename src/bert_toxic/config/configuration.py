from bert_toxic.constants import *
from bert_toxic.utils.common import read_yaml, create_directories
from bert_toxic.entity.entity_config import DataIngestionConfig, DataSplitConfig, SetupModelConfig, TrainingConfig
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

        return DataSplitConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(self.config.data_ingestion.local_data_file),
            train_dir=Path(config.train_dir),
            train_data_path=Path(config.train_data_path),
            test_dir=Path(config.test_dir),
            test_data_path=Path(config.test_data_path),
            params_train_size=self.params.TRAIN_SIZE
        )


    def get_setup_model_config(self) -> SetupModelConfig:
        
        config = self.config.setup_model

        create_directories([config.root_dir])

        return SetupModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            params_classes=self.params.classes,
            params_dropout=self.params.dropout,
        )

    
    def get_training_config(self) -> TrainingConfig:
        
        training = self.config.training
        setup_model = self.config.setup_model

        params = self.params
        
        create_directories([Path(training.root_dir)])

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            model_path=Path(setup_model.model_path),
            trained_model_path=Path(training.trained_model_path),
            training_data_path=Path(self.config.data_split.train_data_path),
            params_max_len=params.max_len,
            params_train_batch_size=params.train_batch_size,
            params_epochs=params.epochs,
            params_learning_rate=params.learning_rate,
            params_train_num_workers=params.train_num_workers,
            params_train_shuffle=params.train_shuffle,
        )



