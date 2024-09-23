import time
from argparse import ArgumentParser, RawTextHelpFormatter
import pandas as pd

from src.preprocess import start_preprocess
from src.train import start_train
from src.evaluate import start_evaluate


def main():
    parser = ArgumentParser(
        description="Main Battery Pipeline", formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "--preprocess_downstream", type=str, default="./data/preprocess/"
    )

    parser.add_argument("--train_downstream", type=str, default="./artifacts/")
    parser.add_argument("--train_history", type=str, default="./artifacts/img/")

    parser.add_argument(
        "--train_epochs",
        type=int,
        default=300,
        help="epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--train_learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )

    parser.add_argument("--evaluate_upstream",
                        type=str,
                        default="./data/preprocess/test/")
    
    parser.add_argument("--evaluate_model_path", 
                        type=str,
                        default="./artifacts/")
    
    parser.add_argument("--evaluate_model_name",
                        type=str,
                        default="eis_model.keras")
    

    args = parser.parse_args()
    preprocess_downstream_directory = args.preprocess_downstream
    train_downstream_directory = args.train_downstream
    train_history_directory = args.train_history
    evaluate_upstream_directory = args.evaluate_upstream
    evaluate_model_path = args.evaluate_model_path
    evaluate_model_name = args.evaluate_model_name

    start_preprocess(downstream_directory=preprocess_downstream_directory)
    start = time.time()
    evaluations = start_train(upstream=preprocess_downstream_directory, downstream=train_downstream_directory, history_directory=train_history_directory,
                epochs=args.train_epochs, batch_size=args.train_batch_size, learning_rate=args.train_learning_rate)
    
    end = time.time()
    
    print("Train Duration: ", end-start)
    pd.DataFrame([evaluations]).to_csv(train_downstream_directory + "result.csv", index=False)

    evaluations = start_evaluate(upstream_directory=evaluate_upstream_directory,
                   model_path=evaluate_model_path,
                   model_name=evaluate_model_name)
    
    for k, v in evaluations.items():
        print(f"metric ({k}) : {v:.4f}")


if __name__ == "__main__":
    main()
