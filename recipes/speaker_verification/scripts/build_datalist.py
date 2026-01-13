#!/usr/bin/env python
# encoding: utf-8

import argparse
import os

import numpy as np
import pandas as pd
import tqdm
from typing import Optional


def findAllSeqs(dirName, extension=".wav", load_data_list=False, speaker_level=1):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers
        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index
        outSpeakers
        The speaker labels (in order)
    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension
    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label.
    Set speaker_label == 0 if no speaker label will be retrieved no matter the
    organization of the dataset.
    """
    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []
    print("finding {}, Waiting...".format(extension))
    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:
            speakerStr = (os.sep).join(root[prefixSize:].split(os.sep)[:speaker_level])
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            for filename in filtered_files:
                full_path = os.path.join(root, filename)
                outSequences.append((speaker, full_path))
    outSpeakers = [None for x in speakersTarget]

    for key, index in speakersTarget.items():
        outSpeakers[index] = key

    print("find {} speakers".format(len(outSpeakers)))
    print("find {} utterance".format(len(outSequences)))

    return outSequences, outSpeakers


def _maybe_add_ext(path: str, extension: str) -> str:
    """If path has no extension, append .{extension}."""
    extension = extension.lstrip(".")
    if os.path.splitext(path)[1] == "":
        return f"{path}.{extension}"
    return path


def _resolve_path(p: str, wav_root: Optional[str]) -> str:
    p = p.strip()
    if wav_root and (not os.path.isabs(p)):
        return os.path.normpath(os.path.join(wav_root, p))
    return p


def convert_veri_test_to_vox1_test(
    veri_test_path: str,
    vox1_test_path: str,
    wav_root: Optional[str] = None,
    extension: str = "wav",
    ensure_extension: bool = True,
) -> None:
    """
    Convert VoxCeleb1 veri_test.txt format:
        <label> <path1> <path2>
    into a new file (vox1_test.txt) with the same 3-column structure,
    optionally prefixing wav_root for relative paths and appending extension.
    """
    n_total = 0
    n_written = 0

    if wav_root is not None:
        wav_root = wav_root.rstrip(os.sep)

    with (
        open(veri_test_path, "r", encoding="utf-8") as fin,
        open(vox1_test_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                # Skip malformed line
                n_total += 1
                continue

            label, p1, p2 = parts[0], parts[1], parts[2]

            if ensure_extension:
                p1 = _maybe_add_ext(p1, extension)
                p2 = _maybe_add_ext(p2, extension)

            p1 = _resolve_path(p1, wav_root)
            p2 = _resolve_path(p2, wav_root)

            fout.write(f"{label} {p1} {p2}\n")
            n_written += 1
            n_total += 1

    print(f"[veri_test -> vox1_test] Read lines: {n_total}, wrote lines: {n_written}")
    print(f"Saved vox1 test file at: {vox1_test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ===== Existing args (train csv building) =====
    parser.add_argument(
        "--extension", help="file extension name", type=str, default="wav"
    )
    parser.add_argument("--dataset_dir", help="dataset dir", type=str, default="data")
    parser.add_argument(
        "--data_list_path", help="list save path", type=str, default="data_list"
    )
    parser.add_argument(
        "--speaker_level", help="speaker label directory level", type=int, default=1
    )

    # ===== New args (veri_test conversion mode) =====
    parser.add_argument(
        "--veri_test_path",
        type=str,
        default=None,
        help="path to veri_test.txt (input). If set, run conversion mode and exit.",
    )
    parser.add_argument(
        "--vox1_test_path",
        type=str,
        default=None,
        help="path to output vox1_test.txt (output). Required in conversion mode.",
    )
    parser.add_argument(
        "--wav_root",
        type=str,
        default=None,
        help="prefix root for relative paths, e.g. /datasets/Voxceleb/vox1/wav",
    )
    parser.add_argument(
        "--no_ensure_extension",
        action="store_true",
        help="do not append extension if a path has no suffix",
    )

    args = parser.parse_args()

    # ===== Conversion mode =====
    if args.veri_test_path is not None:
        if args.vox1_test_path is None:
            raise ValueError("In conversion mode, --vox1_test_path must be provided.")
        convert_veri_test_to_vox1_test(
            veri_test_path=args.veri_test_path,
            vox1_test_path=args.vox1_test_path,
            wav_root=args.wav_root,
            extension=args.extension,
            ensure_extension=(not args.no_ensure_extension),
        )
        raise SystemExit(0)

    # ===== Original behavior: build train csv =====
    outSequences, outSpeakers = findAllSeqs(
        args.dataset_dir,
        extension=args.extension,
        load_data_list=False,
        speaker_level=args.speaker_level,
    )

    outSequences = np.array(outSequences, dtype=str)
    utt_spk_int_labels = outSequences.T[0].astype(int)
    utt_paths = outSequences.T[1]
    utt_spk_str_labels = []
    for i in utt_spk_int_labels:
        utt_spk_str_labels.append(outSpeakers[i])

    csv_dict = {
        "speaker_name": utt_spk_str_labels,
        "utt_paths": utt_paths,
        "utt_spk_int_labels": utt_spk_int_labels,
    }
    df = pd.DataFrame(data=csv_dict)

    try:
        df.to_csv(args.data_list_path, index=False)
        print(f"Saved data list file at {args.data_list_path}")
    except OSError as err:
        print(f"Ran in an error while saving {args.data_list_path}: {err}")
