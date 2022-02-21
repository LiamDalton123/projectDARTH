import logging

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from VAD import VAD
from ConfigVAD import *
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.warning("I'm a warning!")
    logging.info("Hello, Python!")
    logging.debug("I'm a debug message!")

VAD.displayVADConfiguration()
filepath = "CBO_Noise_Speaking_Bionear_Left_Int.wav"
samplerate, data = wavfile.read(filepath)
result_array = np.array([])
buffer = data[0:samplerate]
result = VAD.classifyFrame(buffer, window_size=ConfigVAD.NO_OF_SECONDS * ConfigVAD.FREQUENCY)
result_array = np.append(result_array, result)
logging.info("classifyFrame result = " + str(result))
logging.info(ConfigVAD.PREDICTION[result])
