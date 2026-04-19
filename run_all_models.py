
from training.train_models import TrainModels
import os
import pandas as pd
from datetime import datetime


class RunAllModels:

    def __init__(self, data='hatespeech',test_proportion=0.4,encoder='BERT_CLS'):
        self.data = data
        self.encoder = encoder
        self.models = TrainModels(data=data, test_proportion=test_proportion, encoder=encoder)

    def run_models(self):

        history = {}
        result = {}
        metrics = {}

        ===== Model 1: Simple Model =====
        m, h, r = self.models.train_simple_model()
        metrics['single_model'] = m
        history['single_model'] = h
        result['single_model'] = r

        # ===== Model 2: Multi-Task Model =====
        m, h, r = self.models.train_multi_task_model()
        metrics['multi_task_model'] = m
        history['multi_task_model'] = h
        result['multi_task_model'] = r

        ===== Model 3: One-Hot Model =====
        m, h, r = self.models.train_one_hot_model()
        metrics['one_hot_model'] = m
        history['one_hot_model'] = h
        result['one_hot_model'] = r

        # ===== Model 4: Social Embedding Model =====
        m, h, r = self.models.train_social_embedding_model()
        metrics['social_embedding_model'] = m
        history['social_embedding_model'] = h
        result['social_embedding_model'] = r

        # ===== Model 6: Contrastive Model =====
        m, h, r = self.models.train_contrastive_model(contrastive_loss_w=1)
        metrics['contrastive_model'] = m
        history['contrastive_model'] = h
        result['contrastive_model'] = r

        return metrics, history, result

    def save_results(self):
        

        metrics, result, prediction = self.run_models()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # ============================================================
        # 1. ⭐ Save ALL models' metrics into ONE CSV
        # ============================================================

        metrics_dir = './experiment_results'
        os.makedirs(metrics_dir, exist_ok=True)

        metrics_df = pd.DataFrame(metrics).T

        metrics_filename = os.path.join(metrics_dir, f'{timestamp}_{self.data}_{self.encoder}_metrics.csv')
        metrics_df.to_csv(metrics_filename,index=False)

        # ============================================================
        # 2. Save training history
        # ============================================================
        history_dir = f'./experiment_results/saved_{self.data}_training_history'
        os.makedirs(history_dir, exist_ok=True)

        history_filename = os.path.join(
            history_dir,
            f"{timestamp}_{self.encoder}_training_history.txt"
        )

        with open(history_filename, 'w') as f:
            f.write("TRAINING HISTORY REPORT 0.4-split random_state=36\n")
            f.write("=" * 60 + "\n\n")

            for model_name, model_result in result.items():

                f.write("MODEL INFORMATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Model Type: {model_name}\n\n")

                # ===== Epoch loss =====
                if 'train_loss' in model_result:
                    f.write("DETAILED EPOCH RESULTS:\n")
                    f.write("-" * 30 + "\n")

                    for i, loss in enumerate(model_result['train_loss']):
                        f.write(f"epoch {i+1} -\n")
                        f.write(f"loss: {loss}\n")

                    f.write("\n")

        # ============================================================
        # 3. Save probability predictions
        # ============================================================
        pred_dir = f'./experiment_results/saved_{self.data}_prediction'
        os.makedirs(pred_dir, exist_ok=True)

        pred_filename = os.path.join(
            pred_dir,
            f'{timestamp}_{self.encoder}_probability_predictions.csv'
        )

        pred_df = pd.concat(prediction.values(), axis=1)
        pred_df.to_csv(pred_filename, index=False)

 

if __name__ == "__main__":

    runner = RunAllModels(data='hatespeech',test_proportion=0.3,encoder='SBERT')
    for i in range(6):
        runner.save_results()


    runner = RunAllModels(data='toxic',test_proportion=0.4,encoder='SBERT')
    for i in range(6):
        runner.save_results()








