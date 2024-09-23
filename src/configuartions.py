
from pathlib import Path

PAKAGE_ROOT = Path(__file__)

LOCAL_PATH = PAKAGE_ROOT.resolve().parents[1]

# print(str(LOCAL_PATH / "data" / "processed"))

class DataConfigurations:

    PROCESSED_EIS_TRAIN_FILE = "eis_box_variable.csv"
    PROCESSED_FEATURE_TRAIN_FILE = "feature_box_variable.csv"
    PROCESSED_TRAIN_CELL_INFO = "cell_info_variable.csv"
    
    PROCESSED_EIS_TEST_FILE = "eis_box_fixed.csv"
    PROCESSED_FEATURE_TEST_FILE = "feature_box_fixed.csv"
    PROCESSED_TEST_CELL_INFO = "cell_info_fixed.csv"
    
    LABEL_THRESH_HOLD = 0.8
    PROCESSED_PATH = str(LOCAL_PATH / "data" / "processed")
    PREPROCESS_PATH = str(LOCAL_PATH / "data" / "preprocess")
    
    PREPROCESS_EIS_TRAIN_FILE = "eis_train_dataset.csv"
    PREPROCESS_EIS_TEST_FILE = "eis_test_dataset.csv"
    PREPROCESS_FEATURE_TRAIN_FILE = "feature_train_dataset.csv"
    PREPROCESS_FEATURE_TEST_FILE = "feature_test_dataset.csv"
    

class FeatureConfiguration:
    FEATURES = ["cell_c1_rates", "cell_c2_rates", "cell_d_rates", 'cell_t_charges', 'cell_t1_charges', 'cell_t2_charges', 'cell_ocvs', 'cell_sohs']
    CLASS_NAMES = ["class"]
    REG_NAMES = ["cell_cap_ds"]
    
    
class ModelConfiguration:
    MODEL_PATH = str(LOCAL_PATH / "artifacts")
    IMAEG_PATH = str(LOCAL_PATH / "artifacts" / "img")