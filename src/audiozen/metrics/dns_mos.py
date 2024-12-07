import logging
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import torch

from audiozen.metrics.metric_utils import Metric


options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1

logger = logging.getLogger(__name__)


class pDNSMOS:
    def __init__(self, input_sr=16000) -> None:
        super().__init__()

        root_dir = Path(__file__).parent.absolute()

        self.p835_personal_sess = ort.InferenceSession(
            root_dir / "external" / "pDNSMOS" / "sig_bak_ovr.onnx",
            providers=["CPUExecutionProvider"],
        )

        self.input_sr = input_sr

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40  # type: ignore
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS=False):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio):
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()

        SAMPLERATE = 16000
        INPUT_LENGTH = 9.01

        if self.input_sr != 16000:
            audio = librosa.resample(audio, orig_sr=self.input_sr, target_sr=SAMPLERATE)

        len_samples = int(INPUT_LENGTH * SAMPLERATE)

        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLERATE) - INPUT_LENGTH) + 1

        hop_len_samples = SAMPLERATE
        predicted_p_mos_sig_seg_raw = []
        predicted_p_mos_bak_seg_raw = []
        predicted_p_mos_ovr_seg_raw = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            # p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype("float32")[
            # np.newaxis, :, :
            # ]
            oi = {"input_1": input_features}
            # p808_oi = {"input_1": p808_input_features}
            p_mos_sig_raw, p_mos_bak_raw, p_mos_ovr_raw = self.p835_personal_sess.run(None, oi)[0][0]

            predicted_p_mos_ovr_seg_raw.append(p_mos_ovr_raw)
            predicted_p_mos_sig_seg_raw.append(p_mos_sig_raw)
            predicted_p_mos_bak_seg_raw.append(p_mos_bak_raw)

        clip_dict = {}
        # clip_dict["sr"] = SAMPLERATE
        # clip_dict["len"] = actual_audio_len / SAMPLERATE
        clip_dict["pSIG"] = np.mean(predicted_p_mos_sig_seg_raw)
        clip_dict["pBAK"] = np.mean(predicted_p_mos_bak_seg_raw)
        clip_dict["pOVRL"] = np.mean(predicted_p_mos_ovr_seg_raw)

        return clip_dict


class DNSMOS:
    def __init__(self, input_sr=16000, device=-1) -> None:
        super().__init__()

        root_dir = Path(__file__).parent.absolute()

        # =================== Hack to fix the issue with onnxruntime - Start ===================
        # https://github.com/microsoft/onnxruntime/issues/8313#issuecomment-1486097717
        # Globally modify the default inter_op_num_threads and intra_op_num_threads parameters
        # (Only needed for Slurm system as it has a cgroup)
        _default_session_options = ort.capi._pybind_state.get_default_session_options()

        def get_default_session_options_new():
            _default_session_options.inter_op_num_threads = 1
            _default_session_options.intra_op_num_threads = 1
            return _default_session_options

        ort.capi._pybind_state.get_default_session_options = get_default_session_options_new
        # =================== Hack to fix the issue with onnxruntime - End ===================

        if device > 0:
            print(f"Using device: {device}")
            providers = [("CUDAExecutionProvider", {"device_id": device})]
        else:
            providers = ["CPUExecutionProvider"]

        self.p835_sess = ort.InferenceSession(
            (root_dir / "external" / "DNSMOS" / "sig_bak_ovr.onnx").as_posix(),
            providers=providers,
        )

        self.p808_sess = ort.InferenceSession(
            (root_dir / "external" / "DNSMOS" / "model_v8.onnx").as_posix(),
            providers=providers,
        )

        self.input_sr = input_sr

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40  # type: ignore
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS=False):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio, return_p808=True):
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()

        SAMPLERATE = 16000
        INPUT_LENGTH = 9.01

        if self.input_sr != 16000:
            audio = librosa.resample(audio, orig_sr=self.input_sr, target_sr=SAMPLERATE)

        len_samples = int(INPUT_LENGTH * SAMPLERATE)

        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLERATE) - INPUT_LENGTH) + 1

        hop_len_samples = SAMPLERATE
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            oi = {"input_1": input_features}

            if return_p808:
                p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype("float32")[
                    np.newaxis, :, :
                ]
                p808_oi = {"input_1": p808_input_features}
                p808_mos = self.p808_sess.run(None, p808_oi)[0][0][0]
                predicted_p808_mos.append(p808_mos)

            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.p835_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw)

            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)

        out = [
            Metric(name="OVRL", value=np.mean(predicted_mos_ovr_seg)),
            Metric(name="SIG", value=np.mean(predicted_mos_sig_seg)),
            Metric(name="BAK", value=np.mean(predicted_mos_bak_seg)),
        ]

        if return_p808:
            out.append(Metric(name="p808", value=np.mean(predicted_p808_mos)))
