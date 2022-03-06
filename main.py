import logging
import os

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from datetime import time
from VAD import VAD
from ConfigVAD import *


# data = data / 32768
def displayInfo():
    fileList = os.listdir("test_files/tt_off")
    logging.info("File list: " + str(fileList))
    for file in fileList:
        if file.endswith(".wav") and "_Int" in file:
            filepath = "test_files/tt_off/" + file
            displayFileInfo(filepath)


def displayFileInfo(filepath):
    samplerate, data = wavfile.read(filepath)
    data = data[:30 * samplerate]
    logging.info("File name: " + filepath)
    logging.info("Sample rate: " + str(samplerate))
    logging.info("Length of data: " + str(len(data)))
    gt_array = loadGroundTruthArray(filepath + ".gt", 30)
    plt.figure()
    plt.subplot(311)
    plt.plot(data[:30 * samplerate:1000])
    plt.title = filepath
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(312)
    plt.plot(gt_array)
    plt.xlabel("Time (s)")
    plt.ylabel("Speech")

    plt.subplot(313)
    freqs, times, Sxx = signal.stft(data / 32768, fs=samplerate, window=np.hanning(2048), nfft=2048, nperseg=2048,
                                    noverlap=1024, return_onesided=True)
    plt.pcolor(times, freqs, np.log2(np.abs(Sxx)))  # Sxx as log

    plt.show()


def loadGroundTruthArray(gt_filename, duration_in_seconds):
    gt_array = []
    current_time_in_seconds = 0
    with open(gt_filename) as file:
        lines = file.readlines()
    for line in lines:
        start_time_string, end_time_string = line.strip().split("/")
        logging.info("Start time: " + start_time_string)
        logging.info("End time: " + end_time_string)
        start_time = time.fromisoformat(start_time_string)
        end_time = time.fromisoformat(end_time_string)
        start_time_in_seconds = (start_time.hour * 60 + start_time.minute) * 60 + start_time.second
        end_time_in_seconds = (end_time.hour * 60 + end_time.minute) * 60 + end_time.second
        if end_time.microsecond != 0:
            end_time_in_seconds += 1

        # For time up until speech starts mark it as non-speech (0):
        for second in range(current_time_in_seconds, start_time_in_seconds):
            gt_array.append(0)

        # For time between start and end of speech mark it as speech (1):
        for second in range(start_time_in_seconds, end_time_in_seconds):
            gt_array.append(1)

        current_time_in_seconds = end_time_in_seconds
        print("Ground truth array: " + str(gt_array))

    if duration_in_seconds > end_time_in_seconds:
        for second in range(end_time_in_seconds, duration_in_seconds):
            gt_array.append(0)

    print("Ground truth array: " + str(gt_array))
    return gt_array

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
        buffer = data[startRange:endRange] / 32768
        result = VAD.classifyFrame(buffer, window_size=ConfigVAD.NO_OF_SECONDS * ConfigVAD.FREQUENCY)
        result_array = np.append(result_array, result)
        logging.info("classifyFrame result = " + str(result))
        logging.info(ConfigVAD.PREDICTION[result])
        startRange += samplerate
        endRange += samplerate

    logging.info("Array: " + str(result_array))


def main():
    Tk().withdraw()
    filename = askopenfilename()
    displayFileInfo(filename)
    # categorize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.warning("I'm a warning!")
    logging.info("Hello, Python!")
    logging.debug("I'm a debug message!")
    main()
