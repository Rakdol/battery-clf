import os
import time
import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from logging import getLogger

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    r2_score,
    mean_absolute_error,
)
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import roc_curve

from src.configuartions import DataConfigurations, FeatureConfiguration, ModelConfiguration

logger = getLogger(__name__)


class BatteryDataset(object):
    def __init__(
        self,
        upstream_directory: str,
    ):

        self.upstream_directory = upstream_directory

    def split_real_image(self, eis_data: pd.DataFrame) -> Dict[str, np.array]:
        real_eis = eis_data.iloc[:, 0:100].to_numpy(dtype=np.float32)
        img_eis = eis_data.iloc[:, 100:200].to_numpy(dtype=np.float32)

        return {"real_eis": real_eis, "img_eis": img_eis}

    def split_feature_target(self, feature_data: pd.DataFrame) -> Dict[str, np.array]:
        features = feature_data[FeatureConfiguration.FEATURES].to_numpy(
            dtype=np.float32
        )
        targets = feature_data[FeatureConfiguration.CLASS_NAMES].to_numpy(
            dtype=np.float32
        )

        return {"features": features, "targets": targets}

    def create_tf_dataset(
        self, feature_data: pd.DataFrame, eis_data: pd.DataFrame, trainable=False,
    ) -> tf.data.Dataset:

        feature_target_dict = self.split_feature_target(feature_data)
        real_img_dict = self.split_real_image(eis_data)

        if not trainable:
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    (
                         real_img_dict["real_eis"],
                         real_img_dict["img_eis"],
                         feature_target_dict["features"],
                    ),
                )
            )

            return dataset

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    real_img_dict["real_eis"],
                    real_img_dict["img_eis"],
                    feature_target_dict["features"],
                ),
                feature_target_dict["targets"],
            )
        )
        return dataset

    def split_train_valid(
        self,
        tf_dataset: tf.data.Dataset,
        n_samples: int,
        buffer_size: int,
        batch_size: int,
        split_ratio: float,
        seed: int,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

        tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size, seed=seed)
        tf_dataset = tf_dataset.batch(batch_size)

        train_size = int((1 - split_ratio) * n_samples)
        val_size = n_samples - train_size

        train_dataset = tf_dataset.take(train_size // batch_size)
        valid_dataset = tf_dataset.skip(train_size // batch_size)

        return train_dataset.prefetch(tf.data.AUTOTUNE), valid_dataset.prefetch(
            tf.data.AUTOTUNE
        )

    def pandas_reader_dataset(
        self,
        shuflle_buffer_size=10_000,
        batch_size=32,
        split_ratio=0.2,
        seed=42,
    ) -> Dict[str, tf.data.Dataset]:

        train_path = Path() / self.upstream_directory / "train"
        test_path = Path() / self.upstream_directory / "test"

        feature_train = pd.read_csv(
            str(train_path / DataConfigurations.PREPROCESS_FEATURE_TRAIN_FILE)
        )
        eis_train = pd.read_csv(
            str(train_path / DataConfigurations.PREPROCESS_EIS_TRAIN_FILE)
        )

        train_ds_full = self.create_tf_dataset(
            feature_data=feature_train, eis_data=eis_train, trainable=True,
        )
        
        train_dataset, valid_dataset = self.split_train_valid(
            tf_dataset=train_ds_full,
            n_samples=feature_train.shape[0],
            buffer_size=min(feature_train.shape[0] - 1, shuflle_buffer_size),
            batch_size=batch_size,
            split_ratio=split_ratio,
            seed=seed,
        )

        feature_test = pd.read_csv(
            str(test_path / DataConfigurations.PREPROCESS_FEATURE_TEST_FILE)
        )
        eis_test = pd.read_csv(
            str(test_path / DataConfigurations.PREPROCESS_EIS_TEST_FILE)
        )

        test_dataset = self.create_tf_dataset(
            feature_data=feature_test, eis_data=eis_test
        )
        
        test_dataset = test_dataset.batch(batch_size)

        return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset, "test_target": feature_test["class"].to_numpy(np.int32)}


class BatteryTrainer(object):
    def __init__(
        self,
        train_dataset,
        valid_dataset,
        test_dataset,
        test_target,
        loss,
        optimizer,
        metrics,
        epochs,
        callbacks=None,
    ):

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.test_target = test_target
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.callbacks = callbacks
        self.model = None

    def _build_model(self) -> None:

        # input normalize
        self._normalize()

        real_inputs = tf.keras.layers.Input(shape=(100,))
        img_inputs = tf.keras.layers.Input(shape=(100,))
        feature_inputs = tf.keras.layers.Input(shape=(8,))

        real_norm = self.real_norm_layer(real_inputs)
        real_dense = tf.keras.layers.Dense(128, kernel_initializer="he_uniform")(
            real_norm
        )
        real_batch = tf.keras.layers.BatchNormalization()(real_dense)
        real_elu = tf.keras.layers.ELU()(real_batch)
        real_dropout = tf.keras.layers.Dropout(0.3)(real_elu) 

        img_norm = self.img_norm_layer(img_inputs)
        img_dense = tf.keras.layers.Dense(128, kernel_initializer="he_uniform")(img_norm)
        img_batch = tf.keras.layers.BatchNormalization()(img_dense)
        img_elu = tf.keras.layers.ELU()(img_batch)
        img_dropout = tf.keras.layers.Dropout(0.3)(img_elu) 

        feat_norm = self.feat_norm_layer(feature_inputs)
        feat_dense = tf.keras.layers.Dense(128, kernel_initializer="he_uniform")(
            feat_norm
        )
        feat_batch = tf.keras.layers.BatchNormalization()(feat_dense)
        feat_elu = tf.keras.layers.ELU()(feat_batch)
        feat_dropout = tf.keras.layers.Dropout(0.3)(feat_elu) 

        concat_ = tf.keras.layers.Concatenate()([real_dropout, img_dropout, feat_dropout])
        concat_dense = tf.keras.layers.Dense(128, kernel_initializer="he_uniform")(
            concat_
        )
        concat_batch = tf.keras.layers.BatchNormalization()(
            concat_dense
        )  # BatchNorm 적용
        concat_elu = tf.keras.layers.ReLU()(concat_batch)

        output = tf.keras.layers.Dense(1, activation="sigmoid")(concat_elu)

        self.model = tf.keras.Model(
            inputs=[real_inputs, img_inputs, feature_inputs], outputs=output
        )

        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
        )

    def _normalize(self) -> None:
        # 피처만 추출해서 Normalization 층에 adapt
        self.real_norm_layer = tf.keras.layers.Normalization()
        self.img_norm_layer = tf.keras.layers.Normalization()
        self.feat_norm_layer = tf.keras.layers.Normalization()

        # feature 데이터만 추출해서 adapt 수행
        self.real_norm_layer.adapt(self.train_dataset.map(lambda x, y: x[0]))
        self.img_norm_layer.adapt(self.train_dataset.map(lambda x, y: x[1]))
        self.feat_norm_layer.adapt(self.train_dataset.map(lambda x, y: x[2]))

    def load_model(self, model_path: str):
        if not model_path:
            raise ValueError("Model Path is not provided.")

        if not self.model:
            self._build_model()
            self.model.load_weights(model_path)
        else:
            self.model.load_weights(model_path)

    def save_model(self, model_path: str):
        if not self.model:
            raise ValueError("Model is not built.")

        if self.model:
            self.model.save(model_path)

    def train(self, history_path: Optional[str]):
        if not self.model:
            self._build_model()

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.epochs,
            callbacks=self.callbacks,
        )

        if history_path:
            fig, ax = plt.subplots(1, 2, figsize=(7, 3))

            ax[0].plot(history.history["loss"], label="train_loss")
            ax[0].plot(history.history["val_loss"], label="validation_loss")
            ax[1].plot(history.history["binary_accuracy"], label="train_acc")
            ax[1].plot(history.history["val_binary_accuracy"], label="validation_acc")
            ax[1].plot(history.history["f1_score"], label="train_f1")
            ax[1].plot(history.history["val_f1_score"], label="validation_f1")
            ax[1].legend()

            self.save_fig("training_history")

        return history

    def save_fig(self, fig_id, tight_layout=True, fig_extension="png", resolution=300):

        path = ModelConfiguration.IMAEG_PATH + f"/{fig_id}.{fig_extension}"

        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
        plt.close()

    def evaluate(self) -> Dict[str, float]:

        pred_proba = self.model.predict(self.test_dataset)
        y_preds = np.where(pred_proba > 0.5, 1, 0).reshape(
            -1,
        )
        y_labels = self.test_target

        result = {}

        confusion = confusion_matrix(y_labels, y_preds)

        print("Confusion Matrix")
        print(confusion)

        accuracy = accuracy_score(y_labels, y_preds)
        precision = precision_score(y_labels, y_preds, average="binary")
        recall = recall_score(y_labels, y_preds, average="binary")
        f1 = f1_score(y_labels, y_preds, average="binary")
        roc_auc = roc_auc_score(y_labels, pred_proba)
        print(
            "Accuracy: {0:.4f} %, Precision: {1:.4f} %, Recall: {2:.4f} %,\
             F1_SCORE: {3:.4f} % , AUC:{4:.4f}".format(
                accuracy * 100, precision * 100, recall * 100, f1 * 100, roc_auc
            )
        )

        result["accuary"] = accuracy
        result["precision"] = precision
        result["recall"] = recall
        result["f1"] = f1
        result["roc_auc"] = roc_auc

        return result


class BatteryClassifier(object):
    def __init__(self, model):
        self.model = model
        self.labels = {0: "Normal", 1: "Abnormal"}

    def predict(self, inputs):
        y_proba = self.model.predict(inputs)
        y_preds = np.where(y_proba > 0.5, 1, 0).reshape(
            -1,
        )

        return [p for p in y_preds]
    def predict_proba(self, inputs):
        y_proba = self.model.predict(inputs)

        return [p for p in y_proba]

    def predict_label(self, inputs):
        y_proba = self.model.predict(inputs)
        y_preds = np.where(y_proba > 0.5, 1, 0).reshape(
            -1,
        )
        return [self.labels[p] for p in y_preds]

