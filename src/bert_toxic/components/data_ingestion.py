import os
import urllib.request as request
import gdown
from bert_toxic import logger
from bert_toxic.utils.common import get_size
from bert_toxic.entity.entity_config import DataIngestionConfig

class DataIngestion:
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self) -> str:

        local_file = self.config.local_data_file

        try:
            dataset_url = self.config.source_url
            os.makedirs('artifacts/data_ingestion', exist_ok=True)
            logger.info(f'Downloading data from {dataset_url} into file {local_file}')

            url = 'https://drive.google.com/uc?/export=download&id='+dataset_url.split('/')[-2]
            gdown.download(url, local_file)

            logger.info(f'Downloaded data from {dataset_url} into file {local_file}')

        except Exception as e:
            raise e