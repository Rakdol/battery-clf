import os
import time
import uuid
from argparse import ArgumentParser, RawTextHelpFormatter

import pandas as pd
import tensorflow as tf

from src.models import BatteryDataset, BatteryTrainer, BatteryClassifier


def start_train(
    upstream: str,
    downstream:str,
    history_directory: str,
    epochs:int,
    batch_size:int,
    learning_rate:float,
):

    # Load dataset
    battery_dataset = BatteryDataset(upstream_directory=upstream)
    tf_datasets = battery_dataset.pandas_reader_dataset(batch_size=batch_size)

    # Training Setup
    loss = tf.keras.losses.BinaryCrossentropy()
    optim = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.F1Score(threshold=0.5),
    ]
    early_callback = tf.keras.callbacks.EarlyStopping(patience=50, monitor="val_loss")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
    )
    model_path = downstream + f"eis_model.keras"
    print(f"----------- model_path: {model_path} ------------")

    model_check_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path, monitor="val_loss", mode="min", save_best_only=True
    )
    callbacks = [reduce_lr, early_callback, model_check_callback]

    trainer = BatteryTrainer(
        train_dataset=tf_datasets["train"],
        valid_dataset=tf_datasets["valid"],
        test_dataset=tf_datasets["test"],
        test_target=tf_datasets["test_target"],
        loss=loss,
        optimizer=optim,
        metrics=metrics,
        epochs=epochs,
        callbacks=callbacks,
    )
    history = trainer.train(history_path=history_directory)
    evaluations = trainer.evaluate()

    return evaluations


def main():

    parser = ArgumentParser(
        description="Train battery EIS Classifier Model",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="battery abnormal classficiation",
        help="Battery Classfication using EIS",
    )

    parser.add_argument("--upstream", type=str, default="../data/preprocess/")
    parser.add_argument("--downstream", type=str, default="../artifacts/")
    parser.add_argument("--history", type=str, default="../artifacts/img/")

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )

    args = parser.parse_args()

    upstream = args.upstream
    downstream_directory = args.downstream
    history_directory = args.history

    os.makedirs(downstream_directory, exist_ok=True)
    os.makedirs(history_directory, exist_ok=True)
    start = time.time()
    evaluations = start_train(upstream=upstream, downstream=downstream_directory, history_directory=history_directory, 
                              epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    end = time.time()

    print("Train Duration: ", end-start)
    pd.DataFrame([evaluations]).to_csv(downstream_directory + "result.csv", index=False)


if __name__ == "__main__":
    main()
