# general
PROJECT: str = "disco-instrument-classification"
AUTHOR: str = "satyaborg"
DATASET_NAME: str = "openmic-2018"

# global vars
SEED: int = 42
VERBOSE: int = 1
N_CLASSES: int = 20
N_SAMPLES: int = 20000

# model
MODEL_NAME: str = "vggish"  # model currently in use; options: vggish, yamnet
MODELS: list = dict(
    vggish=dict(in_dim=1280, out_dim=128, url="https://tfhub.dev/google/vggish/1"),
    # yamnet=dict(in_dim=20480, out_dim=1024, url="https://tfhub.dev/google/yamnet/1"),
)

# audio
AUGMENT: bool = False  # set to True to augment the data
TARGET_DURATION: int = 10  # seconds
TARGET_SAMPLE_RATE: str = 16000  # Hz

# hyperparameters
EPOCHS: int = 100
BATCH_SIZE: int = 32  # 32, 64
LEARNING_RATE: float = 1e-04  # 1e-03, 1e-04

# regularizers
DROPOUT: int = 0.5  # 0.5, 0.2
L2_REGULARIZATION: float = 1e-02  # 1e-02, 1e-03

# file paths
ROOT: str = "disco"
DATA_PATH: str = f"{ROOT}/data/raw/{DATASET_NAME}"
LABELS_MAP_PATH: str = f"{DATA_PATH}/class-map.json"
AUDIO_DIR: str = f"{DATA_PATH}/audio"
TARGET_AUDIO_DIR: str = (
    f"{ROOT}/data/processed" if not AUGMENT else f"{ROOT}/data/augmented"
)
PARTITIONS_PATH: str = f"{DATA_PATH}/partitions"
LABELS_PATH: str = f"{DATA_PATH}/openmic-2018-aggregated-labels.csv"
EXPORT_PATH: str = f"{ROOT}/results"
