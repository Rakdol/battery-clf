import os
from typing import Dict, Union, Optional
from pathlib import Path
from logging import getLogger
from argparse import ArgumentParser, RawTextHelpFormatter

import pandas as pd
import numpy as np

from src.configuartions import DataConfigurations


logger = getLogger(__name__)


def retrive_dataset_from_local() -> Dict[str, pd.DataFrame]:
    eis_train = pd.read_csv(
        os.path.join(
            DataConfigurations.PROCESSED_PATH, DataConfigurations.PROCESSED_EIS_TRAIN_FILE
        )
    )
    feature_train = pd.read_csv(
        os.path.join(
            DataConfigurations.PROCESSED_PATH, DataConfigurations.PROCESSED_FEATURE_TRAIN_FILE
        )
    )
    cell_train = pd.read_csv(
        os.path.join(
            DataConfigurations.PROCESSED_PATH, DataConfigurations.PROCESSED_TRAIN_CELL_INFO
        )
    )

    # Drop Null
    feature_train.dropna(inplace=True)
    eis_train.dropna(inplace=True)

    eis_test = pd.read_csv(
        os.path.join(
            DataConfigurations.PROCESSED_PATH, DataConfigurations.PROCESSED_EIS_TEST_FILE
        )
    )
    feature_test = pd.read_csv(
        os.path.join(
            DataConfigurations.PROCESSED_PATH, DataConfigurations.PROCESSED_FEATURE_TEST_FILE
        )
    )

    cell_test = pd.read_csv(
        os.path.join(
            DataConfigurations.PROCESSED_PATH, DataConfigurations.PROCESSED_TEST_CELL_INFO
        )
    )

    # Drop Null
    feature_test.dropna(inplace=True)
    eis_test.dropna(inplace=True)

    return {
        "eis_train": eis_train,
        "feature_train": feature_train,
        "cell_train": cell_train,
        "eis_test": eis_test,
        "feature_test": feature_test,
        "cell_test": cell_test,
    }


def create_label_with_threshold(
    cell_info: pd.DataFrame, feature_data: pd.DataFrame
) -> None:
    classes = pd.Series(dtype=bool)
    
    for cell_id, cell_cap in zip(cell_info.Cell, cell_info.capacity):
        threshold = round(cell_cap * DataConfigurations.LABEL_THRESH_HOLD, 2)
        # print(f"{cell_id}: Label Threshold is {threshold}")
        classes = pd.concat([classes, (feature_data[(feature_data["cell_ids"] == cell_id)]["cell_cap_ds"] > threshold)])
    feature_data["class"] = classes
    feature_data["class"] = feature_data["class"].astype(int)
    print("LABES!!!!!!!!!!!")
    print(feature_data["class"].value_counts())


def save_to_csv(
    data: Union[np.array, pd.DataFrame],
    destination: str,
    name_prefix: str,
    header: Optional[str],
) -> None:
    save_dest = Path(destination)
    filename_format = f"{name_prefix}_dataset.csv"
    csv_path = save_dest / filename_format

    if header:
        df = pd.DataFrame(data, columns=header.split(","))
        df.to_csv(csv_path, index=False)
    else:
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)


def start_preprocess(downstream_directory:str):

    file_path = Path(DataConfigurations.PREPROCESS_PATH + '/train/' + DataConfigurations.PREPROCESS_EIS_TRAIN_FILE)

    # Check if it's a file (and not a directory)
    if file_path.is_file():
        print(f"{file_path} is a file.")
        return 
    else:
        print(f"{file_path} is not a file or does not exist.")


    train_output_destination = os.path.join(downstream_directory, "train")
    test_output_destination = os.path.join(downstream_directory, "test")

    os.makedirs(downstream_directory, exist_ok=True)
    os.makedirs(train_output_destination, exist_ok=True)
    os.makedirs(test_output_destination, exist_ok=True)

    logger.info("Retriving Dataset from local")
    data_dict = retrive_dataset_from_local()
    logger.info("Dataset has been retrived from local")

    # Train: Make label with data configuaration Threshold

    create_label_with_threshold(
        cell_info=data_dict["cell_train"], feature_data=data_dict["feature_train"]
    )
    logger.info("Train label has ben created using threshold")
    # Test Make label with data configuaration Threshold
    create_label_with_threshold(
        cell_info=data_dict["cell_test"], feature_data=data_dict["feature_test"]
    )
    logger.info("Test label has ben created using threshold")

    header_cols = data_dict["feature_train"].columns.to_list()
    header = ",".join(header_cols)

    save_to_csv(
        data_dict["feature_train"], train_output_destination, "feature_train", header=header,
    )
    save_to_csv(data_dict["eis_train"], train_output_destination, "eis_train", header=None)
    save_to_csv(
        data_dict["feature_test"], test_output_destination, "feature_test", header=header,
    )
    save_to_csv(data_dict["eis_test"], test_output_destination, "eis_test", header=None)

def main():
    parser = ArgumentParser(
        description="Make battery EIS dataset", formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "--data",
        type=str,
        default="battery eis",
        help="Battery Classfication using EIS",
    )

    parser.add_argument("--downstream", type=str, default="../data/preprocess/")

    args = parser.parse_args()
    downstream_directory = args.downstream

    start_preprocess(downstream_directory=downstream_directory)


if __name__ == "__main__":
    main()
