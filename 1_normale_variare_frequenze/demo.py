from __future__ import division, print_function

import argparse
import numpy as np
import matplotlib

matplotlib.use('Agg')
import soundfile as sf
import pyworld as pw

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--frame_period", type=float, default=5.0)
parser.add_argument("-s", "--speed", type=int, default=1)

EPSILON = 1e-8


def main(args):
    x, fs = sf.read('voice.wav')
    f0, sp, ap = pw.wav2world(x, fs)

    y = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)
    sf.write('test_f0/y_10_semplice.wav', y, fs)
    sf.write('test_f0+sp/y_10_semplice.wav', y, fs)

    for i in range(1, 20):
        if i != 10:
            _f0 = (i / 10) * np.array(f0)
            _y = pw.synthesize(_f0, sp, ap, fs, args.frame_period)
            sf.write('test_f0/y_' + str(i) + '.wav', _y, fs)

    for i in range(1, 20):
        if i != 10:
            _f0 = (i / 10) * np.array(f0)
            _sp = (i / 10) * np.array(sp)
            _y = pw.synthesize(_f0, _sp, ap, fs, args.frame_period)
            sf.write('test_f0+sp/y_' + str(i) + '.wav', _y, fs)

    print('Please check "test" directory for output files')


if __name__ == '__main__':
    _args = parser.parse_args()
    main(_args)
