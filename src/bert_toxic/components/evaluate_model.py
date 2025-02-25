import torch
import pandas as pd
import numpy as np
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torch import cuda
from sklearn import metrics
from  bert_toxic.entity.entity_config import EvaluationConfig
from bert_toxic.components.setup_model import BertClass
from bert_toxic.components.custom_dataset import CustomDataset
from bert_toxic.utils.common import save_json
from pathlib import Path

class EvaluateModel:

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def setup_device(self):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
    
    def load_model(self):
        self.model = BertClass()
        self.model.load_state_dict(torch.load(self.config.model_path))


    def load_data(self):
        
        df = pd.read_csv(self.config.testing_data_path)
        testing_set = CustomDataset(df, self.config.params_max_len)

        test_params = {
            'batch_size': self.config.params_valid_batch_size,
            'shuffle': self.config.params_valid_shuffle,
            'num_workers': self.config.params_valid_num_workers
        }

        self.testing_loader = DataLoader(testing_set, **test_params)


    def validation(self, epoch):

        self.model.eval()

        fin_targets=[]
        fin_outputs=[]

        with torch.no_grad():
            for _, data in enumerate(self.testing_loader, 0):
            
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)
            
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        return fin_outputs, fin_targets


    def run_validation(self):

        epochs = self.config.params_epochs

        for epoch in range(epochs):
        
            outputs, targets = self.validation(epoch)
            outputs = np.array(outputs) >= 0.5

            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')

            if epoch == epochs-1:
                self.metrics = self.save_metrics(accuracy, f1_score_micro, f1_score_macro)
                self.plot_confusion_matrix(targets, outputs)

            print(f"Accuracy Score = {accuracy}")
            print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")


    @staticmethod
    def plot_confusion_matrix(targets, outputs):
        cm = metrics.confusion_matrix(targets, outputs)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()


    @staticmethod
    def save_metrics(accuracy, f1_micro, f1_macro):
        
        data = {
            'accuracy': accuracy,
            'f1-score-micro': f1_micro,
            'f1-score-macro': f1_macro 
        }
        
        save_json(path=Path('scores.json'), data=data)
        return data


    def log_mlflow(self):
        
        dagshub.init(repo_owner='serize02', repo_name='bert-toxic', mlflow=True)

        with mlflow.start_run():
            
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.metrics)
            mlflow.pytorch.log_model(self.model, 'model', registered_model_name='bert')
            mlflow.pytorch.log_artifact('confusion_matrix.png')
    