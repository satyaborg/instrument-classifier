"""preprocess.py"""
import glob
import time
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import disco.config as config
from disco.scripts.augment import Augmentations
from disco.utils.helpers import load_json, make_dir


print(tf.config.list_physical_devices("GPU"))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize(embedding, label):
    """Serialize the embeddings and label to a tfrecord proto."""
    feature = {
        "feature": _bytes_feature(tf.io.serialize_tensor(embedding)),
        "label": _int64_feature(label),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()


def write_tfrecord(target_path, embeddings, label):
    """Write the embeddings and label to a tfrecord file."""
    with tf.io.TFRecordWriter(target_path) as writer:
        serialized_example = serialize(embeddings.numpy(), label)
        writer.write(serialized_example)


def consolidate_labels(labels_path):
    """Consolidate the labels to the highest relevance instrument per class."""
    agg_labels = pd.read_csv(labels_path)
    agg_labels = (
        agg_labels.groupby("sample_key")
        .agg({"instrument": lambda x: x.iloc[np.argmax(x.values)]})
        .reset_index()
    )
    return agg_labels


def load_label(labels_df, class_map, file_name):
    """Load the label for a given file name."""
    file_name = file_name.split("/")[-1].split(".")[0]
    label = labels_df.loc[labels_df["sample_key"] == file_name, "instrument"].values[0]
    label_index = class_map[label]
    return label_index


def pad_clip(audio, sr, target_duration):
    target_length = int(sr * target_duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    elif len(audio) > target_length:
        audio = audio[:target_length]
    return audio


def load_and_preprocess_audio(
    model, model_name, out_dim, file_path, target_duration=10, target_sr=16000
):
    """Load, resample and feature extract from the audio files."""
    audio, sr = librosa.load(file_path, sr=None, mono=True)  # load audio

    if sr != target_sr:  # resample as needed
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if config.AUGMENT:
        audio = augmentations.apply_augmentations(audio, sr, target_duration)

    else:
        audio = pad_clip(audio, sr, target_duration)

    # forward pass to get the embeddings
    output = model(audio)

    if model_name == "vggish":
        embeddings = output
    elif model_name == "yamnet":
        _, embeddings, log_mel_spectrogram = output

    else:
        raise ValueError("Invalid model name")

    # sanity check
    embeddings.shape.assert_is_compatible_with([None, out_dim])

    return embeddings


if __name__ == "__main__":
    start_time = time.time()

    try:
        for model_name, model_args in config.MODELS.items():
            model_url = model_args.get("url")
            out_dim = model_args.get("out_dim")

            print(f"==> Loading {model_name} from {model_url} ..")
            model = hub.load(model_url)
            print(f"==> Preprocessing audio files from {config.AUDIO_DIR} ..")

            filepaths = glob.glob(
                f"{config.AUDIO_DIR}/*/*.ogg"
            )  # get all the filepaths
            target_filepaths = [
                "/".join(fp.split("/")[-1:]) for fp in filepaths
            ]  # only keep the file name

            print(f"==> Found {len(filepaths)} audio files ..")
            assert len(filepaths) == config.N_SAMPLES

            make_dir(f"{config.TARGET_AUDIO_DIR}/{model_name}_features")

            # load labels
            labels_df = consolidate_labels(config.LABELS_PATH)
            class_map = load_json(config.LABELS_MAP_PATH)

            # intialize augmentations
            augmentations = Augmentations()

            # preprocess each audio file
            for i, fp in enumerate(tqdm(filepaths)):
                target_path = f"{config.TARGET_AUDIO_DIR}/{model_name}_features/{target_filepaths[i]}".replace(
                    ".ogg", ".tfrecord"
                )
                embeddings = load_and_preprocess_audio(
                    model,
                    model_name,
                    out_dim,
                    fp,
                    target_duration=config.TARGET_DURATION,
                    target_sr=config.TARGET_SAMPLE_RATE,
                )
                label = load_label(
                    labels_df=labels_df,
                    class_map=class_map,
                    file_name=target_filepaths[i],
                )
                # np.savez(target_path, embeddings)
                write_tfrecord(target_path, embeddings, label)

            print(
                f"==> Preprocessing completed in {(time.time() - start_time)/60:.2f} mins"
            )

    except Exception as e:
        print(e)
        print("==> Preprocessing failed.")
