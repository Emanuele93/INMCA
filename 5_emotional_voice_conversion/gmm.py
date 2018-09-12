from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames, delta_features
from nnmnkwii.util import apply_each2d_trim
from nnmnkwii.metrics import melcd
from nnmnkwii.baseline.gmm import MLPG
from os.path import basename
from sklearn.mixture import GaussianMixture
from pysptk.synthesis import MLSADF, Synthesizer
import numpy as np
import pyworld
import pysptk
import soundfile as sf

DATA_ROOT = ""

fs = 48000
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
order = 24
frame_period = 5
hop_length = int(fs * (frame_period * 0.001))
num_files = 45
num_test = 3
test_size = 0.03
use_delta = True
show_img = False
emotion_src = "neutral"
emotion_tgt_collection = ["angry", "calm", "disgust", "fearful", "happy", "sad", "surprised"]

if use_delta:
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
else:
    windows = [
        (0, 0, np.array([1.0])),
    ]

vuoto = []
num_vuoto = 25
while num_vuoto > 0:
    vuoto.append(0)
    num_vuoto = num_vuoto - 1


def collect_features(emotion):
    arr = []
    for count in range(0, num_files):
        count_n = count + 1
        path = '_' + str(emotion) + '/' + [str(count_n), ('0' + str(count_n))][count_n < 10] + '.wav'
        x, fs_ = sf.read(path)
        if not isinstance(x[0], float):
            temp = []
            for t in x:
                temp.append(sum(t))
            x = np.array(temp)
        x = x.astype(np.float64)
        f0, time_axis = pyworld.dio(x, fs_, frame_period=frame_period)
        f0 = pyworld.stonemask(x, f0, time_axis, fs_)
        spectrogram = pyworld.cheaptrick(x, f0, time_axis, fs_)
        spectrogram = trim_zeros_frames(spectrogram)
        mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
        mc = mc.tolist()
        while len(mc) < 1000:
            mc.append(vuoto)
        arr.append(mc)
    return np.array(arr)


def test_one_utt(path_src, path_tgt, disable_mlpg=False, diffvc=True):
    if disable_mlpg:
        paramgen = MLPG(gmm, windows=[(0, 0, np.array([1.0]))], diff=diffvc)
    else:
        paramgen = MLPG(gmm, windows=windows, diff=diffvc)

    x, fs_ = sf.read(path_src)
    x = x.astype(np.float64)
    f0, time_axis = pyworld.dio(x, fs_, frame_period=frame_period)
    f0 = pyworld.stonemask(x, f0, time_axis, fs_)
    spectrogram = pyworld.cheaptrick(x, f0, time_axis, fs_)
    aperiodicity = pyworld.d4c(x, f0, time_axis, fs_)

    mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    if use_delta:
        mc = delta_features(mc, windows)
    mc = paramgen.transform(mc)
    if disable_mlpg and mc.shape[-1] != static_dim:
        mc = mc[:, :static_dim]
    assert mc.shape[-1] == static_dim
    mc = np.hstack((c0[:, None], mc))
    if diffvc:
        mc[:, 0] = 0
        engine = Synthesizer(MLSADF(order=order, alpha=alpha), hopsize=hop_length)
        b = pysptk.mc2b(mc.astype(np.float64), alpha=alpha)
        waveform = engine.synthesis(x, b)
    else:
        spectrogram = pysptk.mc2sp(mc.astype(np.float64), alpha=alpha, fftlen=fftlen)
        waveform = pyworld.synthesize(f0, spectrogram, aperiodicity, fs_, frame_period)

    return waveform


X = collect_features(emotion_src)

for emotion_tgt in emotion_tgt_collection:
    Y = collect_features(emotion_tgt)

    X_aligned, Y_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y))
    X_aligned, Y_aligned = X_aligned[:, :, 1:], Y_aligned[:, :, 1:]
    static_dim = X_aligned.shape[-1]
    if use_delta:
        X_aligned = apply_each2d_trim(delta_features, X_aligned, windows)
        Y_aligned = apply_each2d_trim(delta_features, Y_aligned, windows)

    XY = np.concatenate((X_aligned, Y_aligned), axis=-1).reshape(-1, X_aligned.shape[-1] * 2)
    XY = remove_zeros_frames(XY)

    print("\nneutral_" + str(emotion_tgt))

    gmm = GaussianMixture(n_components=8, covariance_type="full", max_iter=100, verbose=1)

    gmm.fit(XY)

    for i in range(num_files, num_files + num_test):
        n = i + 1
        src_path = '_' + str(emotion_src) + '/' + [str(n), ('0' + str(n))][n < 10] + '.wav'
        tgt_path = '_' + str(emotion_tgt) + '/' + [str(n), ('0' + str(n))][n < 10] + '.wav'

        src, _ = sf.read(src_path)
        sf.write(("RIS_neutral_" + str(emotion_tgt) + "\Source_" + basename(src_path)), src, fs)

        tgt, _ = sf.read(tgt_path)
        sf.write(("RIS_neutral_" + str(emotion_tgt) + "\Target_" + basename(tgt_path)), tgt, fs)

        wo_MLPG = test_one_utt(src_path, tgt_path, disable_mlpg=True)
        sf.write("RIS_neutral_" + str(emotion_tgt) + "\wo_MLPG_" + basename(tgt_path), wo_MLPG, fs)

        w_MLPG = test_one_utt(src_path, tgt_path, disable_mlpg=False)
        sf.write("RIS_neutral_" + str(emotion_tgt) + "\w_MLPG_" + basename(tgt_path), w_MLPG, fs)
