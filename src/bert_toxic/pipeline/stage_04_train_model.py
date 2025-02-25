from bert_toxic.config.configuration import ConfigurationManager
from bert_toxic.components.train_model import TrainModel
from bert_toxic import logger

STAGE_NAME = 'training model'

class TrainingPip:

    def __init__(self):
        pass

    def main(self):
        manager = ConfigurationManager()
        config = manager.get_training_config()
        trainer = TrainModel(config=config)
        trainer.setup_device()
        trainer.setup_model()
        trainer.setup_optimizer()
        trainer.load_data()
        trainer.train()


if __name__ == '__main__':
    
    try:
        logger.info(f'Stage {STAGE_NAME} started')
        obj = TrainingPip()
        obj.main()
        logger.info(f'{STAGE_NAME} completed successfully...')
    
    except Exception as e:
        logger.exception(e)
        raise e