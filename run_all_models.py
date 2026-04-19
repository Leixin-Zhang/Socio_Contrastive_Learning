
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

        # ===== Model 1: Simple Model =====
        # m, h, r = self.models.train_simple_model()
        # metrics['single_model'] = m
        # history['single_model'] = h
        # result['single_model'] = r

        # # ===== Model 2: Multi-Task Model =====
        # m, h, r = self.models.train_multi_task_model()
        # metrics['multi_task_model'] = m
        # history['multi_task_model'] = h
        # result['multi_task_model'] = r

        # ===== Model 3: One-Hot Model =====
        # m, h, r = self.models.train_one_hot_model()
        # metrics['one_hot_model'] = m
        # history['one_hot_model'] = h
        # result['one_hot_model'] = r

        # # ===== Model 4: Social Embedding Model =====
        # m, h, r = self.models.train_social_embedding_model()
        # metrics['social_embedding_model'] = m
        # history['social_embedding_model'] = h
        # result['social_embedding_model'] = r

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
    for i in range(5):
        runner.save_results()


    runner = RunAllModels(data='toxic',test_proportion=0.4,encoder='SBERT')
    for i in range(5):
        runner.save_results()



    # runner = RunAllModels(data='hatespeech',test_proportion=0.3,encoder='BERT_CLS')
    # for i in range(8):
    #     runner.save_results()

    # runner = RunAllModels(data='toxic',test_proportion=0.4,encoder='BERT_CLS')
    # for i in range(8):
        # runner.save_results()




    # def save_result(data='toxic'):

    #     if data == 'toxic':
    #         result,prediction = run_toxic_models()

    #     elif data == 'hatespeech':

    #         result,prediction = run_hatespeech_models()

    #     """==================Save training history to a text file==========================="""

    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     history_dir = f'./experiment_results/saved_{data}_training_history'
    #     filename = f"{timestamp}_training_history.txt"
    #     filename = os.path.join(history_dir,filename)
    #     with open(filename, 'w') as f:
    #         f.write("TRAINING HISTORY REPORT 0.4-split random_state=36 \n")
    #         f.write("=" * 60 + "\n\n")
    #         for model_name, result in result.items():
    #             f.write("MODEL INFORMATION:\n")
    #             f.write("-" * 30 + "\n")
    #             f.write(f"Model Type: {model_name}\n")
                
    #             # Detailed epoch results
    #             f.write("DETAILED EPOCH RESULTS:\n")
    #             f.write("-" * 30 + "\n")
    #             for i in range(len(result['train_loss'])):
    #                 f.write(f'epoch {i+1}' "-" + "\n")
    #                 f.write(f'loss {result['train_loss'][i]} \n')

    #                 # ========remove if no validation set=============
    #                 # f.write(f'metrics {result['val_metrics'][i]} \n')


                    
    #     """==================Save prediction to a csv file==========================="""

    #     pred_df = pd.concat(prediction.values(),axis=1)
    #     time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #     pred_dir = f'./experiment_results/saved_{data}_prediction'
    #     pred_filename = f'{time_stamp}_probability_predictions.csv'
    #     pred_filename  = os.path.join(pred_dir,pred_filename)
    #     pred_df.to_csv(pred_filename)


    # if __name__ == "__main__":

    #     # for i in range(5):
    #     #     save_result(data='hatespeech')
    #     #     save_result(data='toxic')
    #     pass



# def run_models(data='toxic', encoder='BERT_CLS'):

#     self.models = TrainModels(data=data, test_proportion=0.3, encoder=encoder)

#     history = {}
#     result = {}
#     metrics = {}

#     # ===== Model 1: Single Model =====
#     h, r, m = self.models.train_simple_model()
#     history['single_model'] = h
#     result['single_model'] = r
#     metrics['single_model'] = m

#     # ===== Model 2: Multi-Task Model =====
#     h, r, m = self.models.train_multi_task_model()
#     history['multi_task_model'] = h
#     result['multi_task_model'] = r
#     metrics['multi_task_model'] = m

#     # ===== Model 3: One-Hot Model =====
#     h, r, m = self.models.train_one_hot_model()
#     history['one_hot_model'] = h
#     result['one_hot_model'] = r
#     metrics['one_hot_model'] = m

#     # ===== Model 4: Social Embedding Model =====
#     h, r, m = self.models.train_social_embedding_model()
#     history['social_embedding_model'] = h
#     result['social_embedding_model'] = r
#     metrics['social_embedding_model'] = m

#     # ===== Model 6: Contrastive Model =====
#     h, r, m = self.models.train_contrastive_model()
#     history['contrastive_model'] = h
#     result['contrastive_model'] = r
#     metrics['contrastive_model'] = m

#     return history, result, metrics


# def run_models(data='toxic',encoder ='BERT_CLS'):


#     self.models = TrainModels(data = data,test_proportion=0.3, encoder= encoder)
#     """
#     Call each model's training function and collect the history and evaluation results.
#     """
#     history = {}
#     result = {}
#     metrics = {}

    
#     # Model 1: Single Lable Model (use aggregated labels for training)
#     single_model_history = self.models.train_simple_model()
#     history['single_model'], result['single_model']= single_model_history

#     # Model 2: Multi-Task Model (Davani paper's method)
#     multi_task_history = self.models.train_multi_task_model()
#     history['multi_task_model'],result['multi_task_model'] = multi_task_history

#     # Model 3: One-Hot Model (one_hot_socio-demo + text embedding)
#     one_hot_history = self.models.train_one_hot_model()
#     history['one_hot_model'],result['one_hot_model'] = one_hot_history


#     # Model 4: Social Embedding Model (socio-embedding + text embedding)
#     social_embedding_history = self.models.train_social_embedding_model()
#     history['social_embedding_model'],result['social_embedding_model'] = social_embedding_history


#     # Model 6: Contrastive Model 
#     contrastive_model_history = self.models.train_contrastive_model()
#     history['contrastive_model'],result['contrastive_model'] = contrastive_model_history



#     return history, result

# def run_hatespeech_models():

#     """
#     Call each model's training function and collect the history and evaluation results.
#     """
#     history = {}
#     result = {}
#     metrics = {}


#     # Model 1: Single Lable Model (use aggregated labels for training)
#     single_model_history = train_hatespeech_models.train_simple_model(epoch=20)
#     history['single_model'], result['single_model']= single_model_history

#     # Model 2: Multi-Task Model
#     multi_task_history = train_hatespeech_models.train_multi_task_model(epoch=9)
#     history['multi_task_model'],result['multi_task_model'] = multi_task_history

#     # Model 3: One-Hot Model
#     one_hot_history = train_hatespeech_models.train_one_hot_model(epoch=5)
#     history['one_hot_model'],result['one_hot_model'] = one_hot_history

#     # Model 4: Social Embedding Model
#     social_embedding_history = train_hatespeech_models.train_social_embedding_model(epoch=6)
#     history['social_embedding_model'],result['social_embedding_model'] = social_embedding_history

#     # Model 6: Contrastive Model
#     contrastive_model_history = train_hatespeech_models.train_contrastive_model(epoch=8)
#     history['contrastive_model'],result['contrastive_model'] = contrastive_model_history

#     return history, result


# if __name__ == "__main__":
#     for i in range(3):
#         run_models()





