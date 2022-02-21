import math

from Matrix import Matrix


class MFCC:
    minMelFreq = 0.0
    maxMelFreq = 4000.0
    lifterExp = 0.6

    def __init__(self, fftSize, numCoeffs, melBands, sampleRate):
        # Precompute mel-scale auditory perceptual spectrum
        self.melWeights = Matrix(melBands, fftSize)

        # Number of non-redundant frequency bins
        self.numFreqs = int(fftSize / 2) + 1
        self.numCoeffs = numCoeffs
        self.melBands = melBands
        self.sampleRate = sampleRate

        fftFreqs = [0.0] * fftSize
        for i in range(0, fftSize):
            fftFreqs[i] = i / fftSize * self.sampleRate

        minMel = self.fhz2mel(self.minMelFreq)
        maxMel = self.fhz2mel(self.maxMelFreq)

        binFreqs = [0.0] * (melBands + 2)
        for i in range(0, melBands + 2):
            binFreqs[i] = self.fmel2hz(minMel + i / (melBands + 1.0) * (maxMel - minMel))

        for i in range(0, melBands):
            for j in range(0, fftSize):
                loSlope = (fftFreqs[j] - binFreqs[i]) / (binFreqs[i + 1] - binFreqs[i])
                hiSlope = (binFreqs[i + 2] - fftFreqs[j]) / (binFreqs[i + 2] - binFreqs[i + 1])
                self.melWeights.A[i][j] = max(0.0, min(loSlope, hiSlope))

        # Keep only positive frequency parts of Fourier transform
        self.melWeights = self.melWeights.getMatrix(0, melBands - 1, 0, self.numFreqs - 1)

        # Precompute DCT matrix
        self.dctMat = Matrix(numCoeffs, melBands)
        scale = math.sqrt(2.0 / melBands)
        for i in range(0, numCoeffs):
            for j in range(0, melBands):
                phase = j * 2 + 1
                self.dctMat.A[i][j] = math.cos(i * phase / (2.0 * melBands) * math.pi) * scale
        root2 = 1.0 / math.sqrt(2.0)
        for j in range(0, melBands):
            self.dctMat.A[0][j] *= root2

        # Precompute lifting vector
        self.lifterWeights = [0.0] * numCoeffs
        self.lifterWeights[0] = 1.0
        for i in range(1, numCoeffs):
            self.lifterWeights[i] = math.pow(i, self.lifterExp)

    def cepstrum(self, re, im):
        powerSpec = Matrix(self.numFreqs, 1)
        for i in range(0, self.numFreqs):
            powerSpec.A[i][0] = re[i] * re[i] + im[i] * im[i]

        # melWeights - melBands x numFreqs
        # powerSpec  - numFreqs x 1
        # melWeights*powerSpec - melBands x 1
        # aSpec      - melBands x 1
        # dctMat     - numCoeffs x melBands
        # dctMat*log(aSpec) - numCoeffs x 1

        aSpec = self.melWeights.times(powerSpec)
        logMelSpec = Matrix(self.melBands, 1)
        for i in range(0, self.melBands):
            logMelSpec.A[i][0] = math.log(aSpec.A[i][0])

        melCeps = self.dctMat.times(logMelSpec)
        ceps = [0.0] * self.numCoeffs
        for i in range(0, self.numCoeffs):
            ceps[i] = self.lifterWeights[i] * melCeps.A[i][0]

        return ceps

    def fmel2hz(self, mel):
        return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0)

    def fhz2mel(self, freq):
        return 2595.0 * math.log10(1.0 + freq / 700.0)
