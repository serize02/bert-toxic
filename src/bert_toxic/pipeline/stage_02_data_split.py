from bert_toxic.config.configuration import ConfigurationManager
from bert_toxic.components.data_split import DataSplit
from bert_toxic import logger

STAGE_NAME = 'data-split'

class DataSplitPip:

    def __init__(self):
        pass

    def main(self):
        manager = ConfigurationManager()
        config = manager.get_data_split_config()
        data_split = DataSplit(config=config)
        data_split.load_data()
        data_split.split()

if __name__ == '__main__':
    
    try:
        logger.info(f'Stage {STAGE_NAME} started')
        obj = DataSplitPip()
        obj.main()
        logger.info(f'{STAGE_NAME} completed successfully...')
    
    except Exception as e:
        logger.exception(e)
        raise e