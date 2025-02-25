from bert_toxic.config.configuration import ConfigurationManager
from bert_toxic.components.evaluate_model import EvaluateModel
from bert_toxic import logger

STAGE_NAME = 'evaluate-model'

class EvaluateModelPip():

    def __init__(self):
        pass


    def main(self):
        manager = ConfigurationManager()
        config = manager.get_evaluation_config()
        evaluation = EvaluateModel(config)
        evaluation.setup_device()
        evaluation.load_model()
        evaluation.load_data()
        logger.info('running evaluation')
        evaluation.run_validation()
        logger.info('tracking to ml-flow server')
        evaluation.log_mlflow()

if __name__ == '__main__':
    
    try:
        logger.info(f'Stage {STAGE_NAME} started')
        obj = EvaluateModelPip()
        obj.main()
        logger.info(f'{STAGE_NAME} completed successfully...')
    
    except Exception as e:
        logger.exception(e)
        raise e