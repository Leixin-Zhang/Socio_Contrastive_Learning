import os
import pandas as pd
from datetime import datetime
from run_all_models import run_hatespeech_models,run_toxic_models



import os
import pandas as pd
from datetime import datetime


def save_result(data='toxic'):

    # ================== Run models ==================
    if data == 'toxic':
        result, prediction = run_toxic_models()

    elif data == 'hatespeech':
        result, prediction = run_hatespeech_models()

    else:
        raise ValueError("data must be 'toxic' or 'hatespeech'")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ============================================================
    # 1. Save training history
    # ============================================================
    history_dir = f'./perspective_modeling/experiment_results/saved_{data}_training_history'
    os.makedirs(history_dir, exist_ok=True)

    history_filename = os.path.join(
        history_dir,
        f"{timestamp}_training_history.txt"
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
    # 2. Save predictions
    # ============================================================
    pred_dir = f'./perspective_modeling/experiment_results/saved_{data}_prediction'
    os.makedirs(pred_dir, exist_ok=True)

    pred_filename = os.path.join(
        pred_dir,
        f'{timestamp}_probability_predictions.csv'
    )

    # prediction = dict of df
    pred_df = pd.concat(prediction.values(), axis=1)
    pred_df.to_csv(pred_filename, index=False)

    # ============================================================
    # 3. ⭐ Save ALL models' metrics into ONE CSV
    # ============================================================
    metrics_list = []

    for model_name, model_result in result.items():

        metrics = model_result.get('metrics', None)

        if metrics is not None:
            row = {
                'model': model_name,
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'f1': metrics.get('f1'),
                'auc': metrics.get('auc')
            }
            metrics_list.append(row)

    metrics_df = pd.DataFrame(metrics_list)

    metrics_dir = f'./perspective_modeling/experiment_results'
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_filename = os.path.join(
        metrics_dir,
        f'{timestamp}_metrics.csv'
    )

    metrics_df.to_csv(metrics_filename, index=False)

    print("✅ Saved files:")
    print(f"- history: {history_filename}")
    print(f"- prediction: {pred_filename}")
    print(f"- metrics: {metrics_filename}")




def save_result(data='toxic'):

    if data == 'toxic':
        result,prediction = run_toxic_models()

    elif data == 'hatespeech':

        result,prediction = run_hatespeech_models()

    """==================Save training history to a text file==========================="""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = f'./perspective_modeling/experiment_results/saved_{data}_training_history'
    filename = f"{timestamp}_training_history.txt"
    filename = os.path.join(history_dir,filename)
    with open(filename, 'w') as f:
        f.write("TRAINING HISTORY REPORT 0.4-split random_state=36 \n")
        f.write("=" * 60 + "\n\n")
        for model_name, result in result.items():
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model Type: {model_name}\n")
            
            # Detailed epoch results
            f.write("DETAILED EPOCH RESULTS:\n")
            f.write("-" * 30 + "\n")
            for i in range(len(result['train_loss'])):
                f.write(f'epoch {i+1}' "-" + "\n")
                f.write(f'loss {result['train_loss'][i]} \n')

                # ========remove if no validation set=============
                # f.write(f'metrics {result['val_metrics'][i]} \n')


                
    """==================Save prediction to a csv file==========================="""

    pred_df = pd.concat(prediction.values(),axis=1)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pred_dir = f'./perspective_modeling/experiment_results/saved_{data}_prediction'
    pred_filename = f'{time_stamp}_probability_predictions.csv'
    pred_filename  = os.path.join(pred_dir,pred_filename)
    pred_df.to_csv(pred_filename)


if __name__ == "__main__":

    # for i in range(5):
    #     save_result(data='hatespeech')
    #     save_result(data='toxic')
    pass