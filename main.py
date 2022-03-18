import logging
from datetime import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
from numpy import size
from scipy import signal
from scipy.io import wavfile

from ConfigVAD import *
from VAD import VAD


def displayFileInfo(filepath):
    samplerate, data = wavfile.read(filepath)
    data = data[:30 * samplerate]
    logging.info("File name: " + filepath)
    logging.info("Sample rate: " + str(samplerate))
    logging.info("Length of data: " + str(len(data)))

    fig, ax = plt.subplots(3, 2)
    plt.subplot(3, 2, 1)
    amplitude_times = np.arange(0, 30, 1000 / samplerate)

    plt.plot(amplitude_times, data[:30 * samplerate:1000])
    plt.title = filepath
    plt.xlabel("Time (s)")
    plt.xlim(0, 30)
    plt.ylabel("Amplitude")

    freqs, times, Sxx = signal.stft(data / 32768, fs=samplerate, window=np.hanning(2048), nfft=2048, nperseg=2048,
                                    noverlap=1024, return_onesided=True)

    plt.subplot(3, 2, (2, 6))
    dB_data = 10 * np.log10(np.abs(Sxx) / np.max(np.abs(Sxx)))
    plt.pcolor(times, freqs, dB_data)  # Sxx as log
    plt.colorbar()

    gt_array = loadGroundTruthArray(filepath + ".gt", 0, 30)
    plt.subplot(3, 2, 3)
    plt.bar(range(0, 29), gt_array, 1.0, color='green')
    # plt.plot(gt_array, color='green')
    plt.xlim(0, 30)
    plt.xlabel("Time (s)")
    plt.ylabel("GT Speech")

    result_array = np.array([])
    rms_array = np.array([])
    time_array = np.array([])

    startRange = 0
    endRange = samplerate

    while endRange < 30 * samplerate:
        buffer = data[startRange:endRange] / 32768
        result = VAD.classifyFrame(buffer, window_size=ConfigVAD.NO_OF_SECONDS * ConfigVAD.FREQUENCY)
        result_array = np.append(result_array, result)
        time_array = np.append(time_array, startRange / samplerate)
        rms_array = np.append(rms_array, np.mean(np.sqrt(buffer ** 2)))
        logging.info("classifyFrame result = " + str(result))
        logging.info(ConfigVAD.PREDICTION[result])
        startRange += samplerate
        endRange += samplerate

    ax[2, 0].bar(range(0, 29), result_array, 1.0, color='blue')
    # Twin the x-axis to make independent y-axes.
    ax[2, 0].twinx().plot(rms_array, color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Speech")
    plt.xlim(0, 30)

    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

    def onrelease(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


def loadGroundTruthArray(gt_filename, start_of_analysis_in_seconds, duration_in_seconds):
    gt_array = np.array([])
    current_time_in_seconds = start_of_analysis_in_seconds
    with open(gt_filename) as file:
        lines = file.readlines()
    for line in lines:
        speech_start_string, speech_end_string = line.strip().split("/")
        logging.info("Start time: " + speech_start_string)
        logging.info("End time: " + speech_end_string)
        speech_start_time = time.fromisoformat(speech_start_string)
        speech_end_time = time.fromisoformat(speech_end_string)
        speech_start_time_in_seconds = (speech_start_time.hour * 60 + speech_start_time.minute) * 60 + speech_start_time.second
        speech_end_time_in_seconds = (speech_end_time.hour * 60 + speech_end_time.minute) * 60 + speech_end_time.second
        if speech_end_time.microsecond != 0:
            speech_end_time_in_seconds += 1

        # don't include ground truth for time before the start of analysis
        if speech_end_time_in_seconds < start_of_analysis_in_seconds:
            continue  # skip the rest of the loop because this bit of ground truth is outside our analysis range

        if speech_start_time_in_seconds < start_of_analysis_in_seconds:
            speech_end_time_in_seconds = start_of_analysis_in_seconds  # skip ground truth between the start of speech to the start of analysis

        # For time up until speech starts mark it as non-speech (0):
        for second in range(current_time_in_seconds, speech_start_time_in_seconds):
            gt_array = np.append(gt_array, 0)

        # For time between start and end of speech mark it as speech (1):
        for second in range(speech_start_time_in_seconds, speech_end_time_in_seconds):
            gt_array = np.append(gt_array, 1)

        current_time_in_seconds = speech_end_time_in_seconds
        print("Ground truth array: " + str(gt_array))

    if duration_in_seconds > speech_end_time_in_seconds:
        for second in range(speech_end_time_in_seconds, duration_in_seconds):
            gt_array = np.append(gt_array, 0)

    print("Ground truth array: " + str(gt_array))
    return gt_array[0:duration_in_seconds - 1]


def categorize():
    VAD.displayVADConfiguration()
    filepath = "test_files/tt_off/CBO_Noise_Speaking_Bionear_Left_Int.wav"
    samplerate, data = wavfile.read(filepath)
    logging.info("Sample rate: " + str(samplerate))
    logging.info("Length of Data: " + str(len(data)))
    result_array = np.array([])
    rms_array = np.array([])
    time_array = np.array([])

    startRange = 0
    endRange = samplerate
    while endRange < len(data):
        buffer = data[startRange:endRange] / 32768
        result = VAD.classifyFrame(buffer, window_size=ConfigVAD.NO_OF_SECONDS * ConfigVAD.FREQUENCY)
        result_array = np.append(result_array, result)
        time_array = np.append(time_array, startRange / samplerate)
        rms_array = np.append(rms_array, np.mean(np.sqrt(buffer ** 2)))
        logging.info("classifyFrame result = " + str(result))
        logging.info(ConfigVAD.PREDICTION[result])
        startRange += samplerate
        endRange += samplerate

    logging.info("Array: " + str(result_array))
    return result_array, time_array, rms_array


def main():
    logging.getLogger('matplotlib.font_manager').disabled = True
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
