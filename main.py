import logging
import os

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from VAD import VAD
from ConfigVAD import *


def displayInfo():
    fileList = os.listdir("test_files/tt_off")
    logging.info("File list: "+str(fileList))
    for file in fileList:
        if file.endswith(".wav"):
            filepath = "test_files/tt_off/"+file
            samplerate, data = wavfile.read(filepath)
            logging.info("File name: "+file)
            logging.info("Sample rate: " + str(samplerate))
            logging.info("Length of Data: " + str(len(data)))
            plt.subplot(100)
            plt.plot(data=data)
            plt.xlabel = "Sample"
            plt.ylabel = "Amplitude"
            plt.show()



def categorize():
    VAD.displayVADConfiguration()
    filepath = "test_files/tt_off/CBO_Noise_Speaking_Bionear_Left_Int.wav"
    samplerate, data = wavfile.read(filepath)
    logging.info("Sample rate: " + str(samplerate))
    logging.info("Length of Data: " + str(len(data)))
    result_array = np.array([])

    startRange = 0
    endRange = samplerate
    while endRange < len(data):
        buffer = data[startRange:endRange]
        result = VAD.classifyFrame(buffer, window_size=ConfigVAD.NO_OF_SECONDS * ConfigVAD.FREQUENCY)
        result_array = np.append(result_array, result)
        logging.info("classifyFrame result = " + str(result))
        logging.info(ConfigVAD.PREDICTION[result])
        startRange += samplerate
        endRange += samplerate

    logging.info("Array: " + str(result_array))


def main():
    displayInfo()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.warning("I'm a warning!")
    logging.info("Hello, Python!")
    logging.debug("I'm a debug message!")
    main()
