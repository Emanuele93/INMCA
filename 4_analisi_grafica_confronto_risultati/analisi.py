from __future__ import division, print_function

import argparse

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import soundfile as sf
import pyworld as pw
import matplotlib.mlab as mlab
import math
import seaborn as sns

from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--frame_period", type=float, default=5.0)
parser.add_argument("-s", "--speed", type=int, default=1)

EPSILON = 1e-8


def savefig(filename, figlist, ylabel="Ampiezza", log=True):
    n = len(figlist)
    f = figlist[0]
    if len(f.shape) == 1:
        plt.figure()
        for i, f in enumerate(figlist):
            plt.subplot(n, 1, i + 1)
            if len(f.shape) == 1:
                plt.plot(f)
                plt.xlim([0, len(f)])
    elif len(f.shape) == 2:
        plt.figure()
        for i, f in enumerate(figlist):
            plt.subplot(n, 1, i + 1)
            if log:
                x = np.log(f + EPSILON)
            else:
                x = f + EPSILON
            plt.imshow(x.T, origin='lower', interpolation='none', aspect='auto', extent=(0, x.shape[0], 0, x.shape[1]))
    else:
        raise ValueError('Input dimension must < 3.')
    plt.xlabel("Tempo")
    plt.ylabel(ylabel)
    plt.savefig(filename)


def saveGmmFig(samples, name):
    plt.clf()
    mu, variance = norm.fit(samples[0])
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
    plt.plot(x, mlab.normpdf(x, mu, sigma), 'C1', label='src')
    mu, variance = norm.fit(samples[1])
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
    plt.plot(x, mlab.normpdf(x, mu, sigma), 'C2', label='tgt')
    mu, variance = norm.fit(samples[2])
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
    plt.plot(x, mlab.normpdf(x, mu, sigma), 'C3', label='obt')
    plt.legend()
    plt.xlabel("Frequenza")
    plt.ylabel("Densità")
    plt.savefig('analisi/G_' + name + '.png')

    plt.clf()
    sns.distplot(samples[0], 'C1', label='src', hist=False)
    sns.distplot(samples[1], 'C2', label='tgt', hist=False)
    sns.distplot(samples[2], 'C3', label='obt', hist=False)
    plt.xlabel("Frequenza")
    plt.ylabel("Densità")
    plt.savefig('analisi/GMM_' + name + '.png')


def drawGmm(_f0, _sp):
    samples = [[], [], []]
    for e in range(0, 3):
        for i in range(len(_f0[e])):
            if _f0[e][i] != 0:
                samples[e].append(_f0[e][i])
    saveGmmFig(samples, "f0")

    samples = [[], [], []]
    for e in range(0, 3):
        for i in _sp[e]:
            for j in range(0, len(i)):
                if int(i[j]*1000) != 0:
                    for k in range(0, int(i[j]*1000)):
                        samples[e].append(j)
    saveGmmFig(samples, "sp")


_x_src, fs_src = sf.read('voice_src.wav')
f0_src, sp_src, ap_src = pw.wav2world(_x_src, fs_src)
y_src = pw.synthesize(f0_src, sp_src, ap_src, fs_src, pw.default_frame_period)

_x_tgt, fs_tgt = sf.read('voice_tgt.wav')
f0_tgt, sp_tgt, ap_tgt = pw.wav2world(_x_tgt, fs_tgt)
y_tgt = pw.synthesize(f0_tgt, sp_tgt, ap_tgt, fs_tgt, pw.default_frame_period)

_x_obt, fs_obt = sf.read('voice_obt.wav')
f0_obt, sp_obt, ap_obt = pw.wav2world(_x_obt, fs_obt)
y_obt = pw.synthesize(f0_obt, sp_obt, ap_obt, fs_obt, pw.default_frame_period)

savefig('analisi/wavform.png', [y_src, y_tgt, y_obt])
savefig('analisi/f0.png', [f0_src, f0_tgt, f0_obt], ylabel="Frequenza")
savefig('analisi/sp.png', [sp_src, sp_tgt, sp_obt], ylabel="Frequenza")

drawGmm([f0_src, f0_tgt, f0_obt], [sp_src, sp_tgt, sp_obt])

print('Please check "analisi" directory for output files')
