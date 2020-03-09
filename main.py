from copy import copy
from sys import argv
from warnings import filterwarnings

from numpy import mean, argmax, log, arange
from numpy.fft import rfft
from numpy.random import seed
from scipy import signal
from scipy.signal import decimate, windows
from soundfile import read

seed(136749 + 136714)
filterwarnings("ignore")


def compute_frequency(sig, fs):
    N = len(sig)
    windowed = sig * windows.flattop(N, sym=False)
    X = log(abs(rfft(windowed)))

    hps = copy(X)
    for h in arange(2, 9):
        dec = decimate(X, h)
        hps[:len(dec)] += dec
    i_peak = argmax(hps)

    return fs * i_peak / N


def high_pass_filter(sig):
    sos = signal.butter(2, 8, 'hp', fs=800, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered


if __name__ == "__main__":
    # noinspection PyBroadException
    try:
        audio, fs = read(argv[1])

        if len(audio.shape) > 1:
            audio = mean([audio[:, 0], audio[:, 1]], axis=0)

        time_seconds = arange(audio.size, dtype=float) / fs

        if 2 > time_seconds[-1]:
            time_start = 0.5
        else:
            time_start = time_seconds[0]

        if 3 < time_seconds[-1]:
            time_end = time_seconds[-1] - 0.5
        else:
            time_end = time_seconds[-1]

        audio = audio[(time_seconds >= time_start) & (time_seconds <= time_end)]

        audio = high_pass_filter(audio)
        hz = compute_frequency(audio, fs)

        if hz < 170:
            print("M")
        else:
            print("K")
    except:
        print("M")
