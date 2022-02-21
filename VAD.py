import time
from Classifier import *
from FeatureExtractor import *
import logging

TAG = "VAD"
log = logging.getLogger(TAG)


class VAD:

    # Displays configuration of VAD

    @staticmethod
    def displayVADConfiguration():
        log.debug("VAD Parameters")
        log.debug("FRAME_SIZE_MS: " + str(ConfigVAD.FRAME_SIZE_MS))
        log.debug("NO_OF_SECONDS: " + str(ConfigVAD.NO_OF_SECONDS))
        log.debug("FREQUENCY: " + str(ConfigVAD.FREQUENCY))
        log.debug("SAMPLES_PER_FRAME: " + str(ConfigVAD.SAMPLES_PER_FRAME))
        log.debug("CLASSIFIC_DURATION_MS: " + str(ConfigVAD.CLASSIFICATION_DURATION_MS))
        log.debug("NO_OF_WIN_PER_DURATION: " + str(ConfigVAD.NO_OF_WINDOWS_PER_DURATION))
        log.debug("VOICE_THRESHOLD: " + str(ConfigVAD.VOICE_THRESHOLD))
        log.debug("RMS_THRESHOLD: " + str(ConfigVAD.RMS_THRESHOLD))

    # Classifies all frames
    #      * @param buffer
    #      * @param window_size
    #      * @return classification

    @staticmethod
    def classifyFrame(buffer, window_size):
        classification = 0
        voiced = 0  # counts number of speech classifications in window

        # Extract features and classify for each frame
        for k in range(0, window_size, ConfigVAD.SAMPLES_PER_FRAME):
            if k == 0:
                log.info(TAG + "About to extract and classify first frame..." + str(round(time.time() * 1000)))

            features = FeatureExtractor.ComputeFeaturesForFrame(buffer, ConfigVAD.SAMPLES_PER_FRAME, k)
            # Classify sample
            if Classifier.Classify(features):
                voiced += 1

            if k == 0:
                log.info(TAG + "Done classifying first sample..." + str(round(time.time() * 1000)))

        voiceCount = voiced

        # Check if number of samples classified as voiced is greater than threshold
        if voiced >= ConfigVAD.VOICE_THRESHOLD:
            classification = 1

        if ConfigVAD.DEBUG_MODE:
            log.info("Voice Count: " + str(voiced))

        return classification

    # Checks and returns if sound sample is silence
    #      * @param buffer
    #      * @return isSilence

    @staticmethod
    def isSilence(buffer):
        isSilence = True

        # Calculate energy
        energy = VAD.calculateRMS(buffer)

        # Check if above threshold
        if energy > ConfigVAD.RMS_THRESHOLD:
            isSilence = False

        log.info("RMS: " + str(energy))
        log.debug("Silence: " + str(isSilence))
        return isSilence

    # Estimate the RMS of the sound sample
    #      * @param buffer
    #      * @return

    @staticmethod
    def calculateRMS(buffer):
        minimum = 1     # Will be recalculated in the for loop
        minRaw = 32767
        maxRaw = -32768
        maximum = -1    # Will be recalculated in the for loop
        meanAbsolute = 0
        sumAbsolute = 0
        energy = 0
        mappedSample = 0
        minIndex = -1
        i = 0
        length = 0

        for sample in range(buffer):
            mappedSample = sample / abs(maxRaw)

            if abs(mappedSample) > ConfigVAD.DEVICE_NOISE_LEVEL:
                energy += mappedSample*mappedSample
                sumAbsolute += abs(mappedSample)

                if mappedSample < minimum:
                    minIndex = i

                minimum = min(minimum, mappedSample)
                maximum = max(maximum, mappedSample)

                minRaw = min(minRaw, sample)

                length += 1

            i += 1

        if length == 0:
            return 0

        meanAbsolute = sumAbsolute / length
        rms = math.sqrt(energy/length)

        if ConfigVAD.DEBUG_MODE:
            log.debug("No. of samples: " + str(i))
            log.debug("RMS: " + str(rms))
            log.debug("Max: " + str(maximum))
            log.debug("Min: " + str(minimum))
            log.debug("Min Raw: " + str(minRaw))
            log.debug("Min Index: " + str(minIndex))
            log.debug("Mean Absolute: " + str(meanAbsolute))

        return rms


