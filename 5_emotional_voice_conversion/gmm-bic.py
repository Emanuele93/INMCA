from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames, delta_features
from nnmnkwii.util import apply_each2d_trim
from nnmnkwii.metrics import melcd
import numpy as np
import pyworld
import pysptk
import soundfile as sf
from sklearn import mixture

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
emotion_tgt_collection = ["angry"]#, "calm", "disgust", "fearful", "happy", "sad", "surprised"]

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

    lowest_bic = np.infty
    lowest_ncomp = 0
    bic = []
    n_components_range = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type="full", max_iter=100, verbose=1)
        gmm.fit(XY)
        bic.append(gmm.bic(XY))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
            lowest_ncomp = n_components

    print("numero di componenti ottimale: " + str(lowest_ncomp))
