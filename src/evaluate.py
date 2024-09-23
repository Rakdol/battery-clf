import os
from argparse import ArgumentParser, RawTextHelpFormatter

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import roc_curve

from src.models import BatteryClassifier
from src.configuartions import DataConfigurations, FeatureConfiguration



def create_test_dataset(feature_path:str,
                        eis_path:str):
    feature_test = pd.read_csv(feature_path)
    eis_test = pd.read_csv(eis_path)

    real_eis = eis_test.iloc[:, 0:100].to_numpy(dtype=np.float32)
    img_eis = eis_test.iloc[:, 100:200].to_numpy(dtype=np.float32)
    features = feature_test[FeatureConfiguration.FEATURES].to_numpy(
            dtype=np.float32
        )
    targets = feature_test[FeatureConfiguration.CLASS_NAMES].to_numpy(
            dtype=np.float32
        )
    
    return {"X": [real_eis, img_eis, features], "y": targets}

def save_confusion_matrix_fig(y_labels, y_preds, img_path:str):
    cm = confusion_matrix(y_labels, y_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the figure
    plt.savefig(img_path)  # Save as PNG


def start_evaluate(upstream_directory:str,
                   model_path:str, 
                   model_name:str):
    
    feature_path = upstream_directory + DataConfigurations.PREPROCESS_FEATURE_TEST_FILE
    eis_path = upstream_directory + DataConfigurations.PREPROCESS_EIS_TEST_FILE

    data = create_test_dataset(feature_path, eis_path)
    model = tf.keras.saving.load_model(model_path + model_name)
    classifier = BatteryClassifier(model)


    evaluate_distnination = os.path.join(model_path, "evaluate")
    os.makedirs(evaluate_distnination, exist_ok=True)

    y_pred_proba = classifier.predict_proba(data['X'])
    y_preds = classifier.predict(data['X'])
    y_labels = data['y']

    save_confusion_matrix_fig(y_labels=y_labels, 
                              y_preds=y_preds, 
                              img_path=os.path.join(evaluate_distnination, "confusion.png"))

    evaluations = {}

    accuracy = accuracy_score(y_labels, y_preds)
    precision = precision_score(y_labels, y_preds, average="binary")
    recall = recall_score(y_labels, y_preds, average="binary")
    f1 = f1_score(y_labels, y_preds, average="binary")
    roc_auc = roc_auc_score(y_labels, y_pred_proba)
    
    evaluations["accuary"] = accuracy
    evaluations["precision"] = precision
    evaluations["recall"] = recall
    evaluations["f1"] = f1
    evaluations["roc_auc"] = roc_auc

    pd.DataFrame([evaluations]).to_csv(os.path.join(evaluate_distnination, "eval_result.csv"), index=False)

    return evaluations


def main():
    parser = ArgumentParser(
        description="Evaluate trained Model", formatter_class=RawTextHelpFormatter,
    )
    
    parser.add_argument("--upstream",
                        type=str,
                        default="../data/preprocess/test/")
    
    parser.add_argument("--model_path", 
                        type=str,
                        default="../artifacts/")
    
    parser.add_argument("--model_name",
                        type=str,
                        default="eis_model.keras")
    
    
    args = parser.parse_args()

    upstream_directory = args.upstream
    model_path = args.model_path
    model_name = args.model_name


    evaluations = start_evaluate(upstream_directory=upstream_directory,
                   model_path=model_path,
                   model_name=model_name)
    
    print(evaluations)


if __name__ == "__main__":
    main()