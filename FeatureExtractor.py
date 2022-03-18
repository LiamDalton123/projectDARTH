from ConfigVAD import *
from FFT import *
from MFCC import *
from Window import *


class FeatureExtractor:
    FFT_SIZE = 2048
    BITRATE = 36000
    MFCCS_VALUE = 13
    MEL_BANDS = 26

    featureFFT = FFT(FFT_SIZE)
    featureWin = Window(ConfigVAD.SAMPLES_PER_FRAME)
    featureMFCC = MFCC(FFT_SIZE, MFCCS_VALUE, MEL_BANDS, BITRATE)

    @staticmethod
    def ComputeFeaturesForFrame(data16bit, size, index):
        fftBufferR = [0] * FeatureExtractor.FFT_SIZE
        fftBufferI = [0] * FeatureExtractor.FFT_SIZE

        # convert audio buffer to doubles
        for i in range(0, size):
            fftBufferR[i] = data16bit[index + i]

        # In-place windowing
        FeatureExtractor.featureWin.applyWindow(fftBufferR)

        # In-place FFT
        FeatureExtractor.featureFFT.fft(fftBufferR, fftBufferI)

        # Get MFCCs
        featureCepstrum = FeatureExtractor.featureMFCC.cepstrum(fftBufferR, fftBufferI)

        return featureCepstrum
