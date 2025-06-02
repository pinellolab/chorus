# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kipoiseq",
#     "numpy",
#     "pandas",
#     "pyfaidx",
#     "tensorflow",
#     "tensorflow-hub",
#     "tqdm",
# ]
# ///
import json
import os
from dataclasses import dataclass

import kipoiseq
import numpy as np
import pandas as pd
import pyfaidx
import tensorflow as tf
import tensorflow_hub as hub
from kipoiseq import Interval
from tqdm import tqdm


os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"


class Enformer:
    def __init__(self, tfhub_url: str):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence, target_mask, output_head="human"):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
        prediction = (
            tf.reduce_sum(target_mask[tf.newaxis] * self._model.predict_on_batch(input_sequence)[output_head])
            / target_mask_mass
        )

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)


class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(
            interval.chrom,
            max(interval.start, 0),
            min(interval.end, chromosome_length),
        )
        # pyfaidx wants a 1-based interval
        sequence = str(
            self.fasta.get_seq(trimmed_interval.chrom, trimmed_interval.start + 1, trimmed_interval.stop).seq
        ).upper()
        # Fill truncated values with N's.
        pad_upstream = "N" * max(-interval.start, 0)
        pad_downstream = "N" * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


@dataclass
class EnformerBase:
    model: Enformer
    track_dict: list
    sequences: pd.DataFrame


def create_enformer_base(
    model_path: str,
    sequence_path: str,
    tag_filter: list | None = None,
) -> EnformerBase:
    model = Enformer(model_path)
    sequences = pd.read_csv(sequence_path, sep="\t")
    if tag_filter:
        sequences = sequences[sequences["TAG"].isin(tag_filter)]
    with open("data/enformer/tracks_list_all.json") as f:
        tracks_dict = json.load(f)
    return EnformerBase(model, tracks_dict, sequences)


def load_region_from_csv(file_path: str) -> list[str | int]:
    df = pd.read_csv(file_path, sep="\t", header=None)
    return [df.iloc[0][0], df.iloc[0][1], df.iloc[0][2]]


def create_construct_sequences(
    sequences_df: pd.DataFrame,
    sequence_col: str,
    target_interval: kipoiseq.Interval,
    seq_extractor: FastaStringExtractor,
    sequence_length: int,
):
    # Extract the common sequence just once
    template = seq_extractor.extract(target_interval.resize(sequence_length))
    center = len(template) // 2
    prefix = template[: center - 100]
    suffix = template[center + 100 :]

    return [prefix + seq + suffix for seq in sequences_df[sequence_col]]


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def predict_from_sequence(model: Enformer, input_sequence: str) -> np.ndarray:
    sequence_one_hot = one_hot_encode(input_sequence)
    return model.predict_on_batch(sequence_one_hot[np.newaxis])["human"][0]


def generate_predictions_nobw(
    sequence: str,
    tracks: dict,
    model: Enformer,
    span: int = 128,
) -> np.ndarray:
    predictions = predict_from_sequence(model, sequence)
    track_ids = np.array([track["id"] for track in tracks])
    selected_predictions = predictions[:, track_ids]
    mod_predictions = np.repeat(selected_predictions, span, axis=0)
    return mod_predictions


def run_predictions_replace_nobw(
    fasta_path: str,
    enformer: EnformerBase,
    output_path: str,
    enhancer_region: str,
    cage_region: str,
    h3k4me3_region: str,
) -> None:
    fasta_extractor = FastaStringExtractor(fasta_path)
    enhancer_region = load_region_from_csv(enhancer_region)
    cage_region = load_region_from_csv(cage_region)
    h3k4me3_region = load_region_from_csv(h3k4me3_region)

    target_interval = kipoiseq.Interval(enhancer_region[0], int(enhancer_region[1]), int(enhancer_region[2]))

    # Loading input sequences
    sequences = enformer.sequences
    sequences["CONSTRUCT_SEQ"] = create_construct_sequences(
        sequences, "SEQUENCE", target_interval, fasta_extractor, 393216
    )

    sequences = sequences[["SEQUENCE", "CONSTRUCT_SEQ", "SEQ_ID", "CELL_TYPE", "TAG"]].values.tolist()

    enh_values = []
    cage_values = []
    h3k4me3_values = []

    track_names = [x["name"] for x in enformer.track_dict]

    for i, (sequence, construct_seq, id, cell_type, tag) in tqdm(enumerate(sequences), total=len(sequences)):
        predictions = generate_predictions_nobw(
            sequence=construct_seq,
            tracks=enformer.track_dict,
            model=enformer.model,
        )
        # n x 896 * 128 = n x 114688

        mod_start = int(enhancer_region[1] + ((enhancer_region[2] - enhancer_region[1]) / 2)) - int(114688 / 2)
        enh_start = np.abs(mod_start - enhancer_region[1])
        enh_end = enh_start + (enhancer_region[2] - enhancer_region[1])

        promoter_start = np.abs(mod_start - cage_region[1])
        promoter_end = promoter_start + (cage_region[2] - cage_region[1])

        # Mean preds for enhancer region where ouput includes shape (114688, track_num)
        enh_preds = np.mean(predictions[enh_start:enh_end], axis=0)
        enh_preds = pd.Series(enh_preds, index=track_names)
        enh_preds["SEQ_ID"] = id
        enh_preds["CELL_TYPE"] = cell_type
        enh_preds["TAG"] = tag
        enh_preds["SEQUENCE"] = sequence
        enh_values.append(enh_preds)

        cage_preds = np.mean(predictions[promoter_start:promoter_end], axis=0)
        cage_preds = pd.Series(cage_preds, index=track_names)
        cage_values.append(cage_preds)

        # Mean preds for h3k4me3 region where ouput includes shape (114688, track_num)
        h3k4me3_preds = np.mean(predictions[promoter_start:promoter_end], axis=0)
        h3k4me3_preds = pd.Series(h3k4me3_preds, index=track_names)
        h3k4me3_values.append(h3k4me3_preds)

    df_out_ENH = pd.DataFrame(
        [x.values.tolist() for x in enh_values], columns=["ENHANCER_" + x for x in enh_preds.index]
    )
    # Only keep preds that contain DNASE in the column name
    df_out_ENH = df_out_ENH[
        df_out_ENH.columns[df_out_ENH.columns.str.contains("DNASE|SEQ_ID|CELL_TYPE|CAGE|SEQUENCE|TAG", regex=True)]
    ]
    cage_cols = df_out_ENH.columns[df_out_ENH.columns.str.contains("CAGE")]
    # Rename only the CAGE columns
    df_out_ENH.rename(columns={c: c.replace("ENHANCER_", "enhancer_") for c in cage_cols}, inplace=True)

    df_out_CAGE = pd.DataFrame(
        [x.values.tolist() for x in cage_values], columns=["cGENE_" + x for x in cage_preds.index]
    )
    # Only keep preds that contain CAGE in the column name
    df_out_CAGE = df_out_CAGE[df_out_CAGE.columns[df_out_CAGE.columns.str.contains("CAGE", regex=True)]]

    df_out_H3K4ME3 = pd.DataFrame(
        [x.values.tolist() for x in h3k4me3_values],
        columns=["hGENE_" + x for x in h3k4me3_preds.index],
    )
    # Only keep preds that contain H3K4ME3 in the column name
    df_out_H3K4ME3 = df_out_H3K4ME3[df_out_H3K4ME3.columns[df_out_H3K4ME3.columns.str.contains("H3K4ME3")]]
    df_out = pd.concat([df_out_ENH, df_out_CAGE, df_out_H3K4ME3], axis=1)

    # Remove ENHANCER_, cGENE_, hGENE_ from column names
    df_out.columns = df_out.columns.str.replace("ENHANCER_", "")
    df_out.columns = df_out.columns.str.replace("cGENE_", "")
    df_out.columns = df_out.columns.str.replace("hGENE_", "")

    print("Saving output to data/outputs/enformer/enformer_predictions.txt")
    df_out.to_csv(f"{output_path}/enformer_predictions.txt", sep="\t", index=False)

if __name__ == "__main__":
    enformer_base = create_enformer_base(
        model_path="https://tfhub.dev/deepmind/enformer/1",
        sequence_path="data/enformer/sequences.tsv",
        tag_filter=["tag1", "tag2"],
    )
    run_predictions_replace_nobw(
        fasta_path="data/genome/hg38.fa",
        enformer=enformer_base,
        output_path="data/outputs/enformer",
        enhancer_region="data/enhancer_region.bed",
        cage_region="data/cage_region.bed",
        h3k4me3_region="data/h3k4me3_region.bed",
    )
