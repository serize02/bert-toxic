from bert_toxic.config.configuration import ConfigurationManager
from bert_toxic.components.data_ingestion import DataIngestion
from bert_toxic import logger

STAGE_NAME = 'data-ingestion'

class DataIngestionPip:

    def __init__(self):
        pass

    def main(self):

        manager = ConfigurationManager()
        config = manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=config)
        data_ingestion.download_file()


if __name__ == '__main__':
    
    try:
        logger.info(f'Stage {STAGE_NAME} started')
        obj = DataIngestionPip()
        obj.main()
        logger.info(f'{STAGE_NAME} completed successfully...')
    
    except Exception as e:
        logger.exception(e)
        raise e
        
