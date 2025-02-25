from bert_toxic.config.configuration import ConfigurationManager
from bert_toxic.components.setup_model import SetupModel
from bert_toxic import logger

STAGE_NAME = 'setup-model'

class SetupModelTrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        manager = ConfigurationManager()
        config = manager.get_setup_model_config()
        setup_model = SetupModel(config)
        setup_model.get_model()
        

if __name__ == '__main__':
    try:
        logger.info(f'Stage {STAGE_NAME} started')
        obj = SetupModelTrainingPipeline()
        obj.main()
        logger.info(f'{STAGE_NAME} completed successfully...')
    
    except Exception as e:
        logger.exception(e)
        raise e