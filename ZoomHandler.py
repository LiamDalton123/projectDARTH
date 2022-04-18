import logging
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib.widgets import Button
from scipy import signal
from scipy.io import wavfile

from GroundTruthReader import GroundTruthReader
from VADLite.ConfigVAD import *
from VADLite.VAD import VAD
from VADLiteAdapter import VADLiteAdapter


class ZoomHandler:

    def __init__(self):
        self.start_time_sec = None
        self.end_time_sec = None
        self.data = None
        self.samplerate = None

    def display_file_info(self, filepath):
        self.samplerate, self.data = wavfile.read(filepath)

        self.start_time_sec = 0
        self.end_time_sec = 30
        # zero out what was before start time (to keep clocks right), and ignore what was  after end time.
        logging.info("File name: " + filepath)
        logging.info("Sample rate: " + str(self.samplerate))
        logging.info("Length of data: " + str(len(self.data)))

        self.fig, self.ax = plt.subplots()
        plt.title(filepath)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        plt.subplots_adjust(bottom=0.2)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        logging.info("cid=" + str(cid))
        cid = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        logging.info("cid=" + str(cid))
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.ondrag)
        logging.info("cid=" + str(cid))

        # buttons
        self.axZoom = plt.axes([0.65, 0.02, 0.1, 0.070])
        self.zoomButton = Button(self.axZoom, 'Zoom')
        self.axCancel = plt.axes([0.8, 0.02, 0.1, 0.070])
        self.cancelButton = Button(self.axCancel, 'Cancel', color='#90EE90')

        self.display_data()

        plt.ion()
        plt.show(block=True)

    def display_data(self):
        start_tick = math.ceil(self.start_time_sec * self.samplerate)
        end_tick = math.floor(self.end_time_sec * self.samplerate)
        analysis_buffer = numpy.concatenate([[0] * start_tick, self.data[start_tick: end_tick]])
        amplitude_times = np.arange(self.start_time_sec, self.end_time_sec, 1000 / self.samplerate)
        self.ax.plot(amplitude_times, analysis_buffer[start_tick:end_tick:1000])
        self.ax.set_xlim(self.start_time_sec, self.end_time_sec)

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        allaxes = self.fig.get_axes()
        allaxes[0].axvline(x=event.xdata, visible=True, color="green", zorder=-100)
        self.clickstart = event.xdata
        self.lastDragPosition = event.xdata
        self.fig.canvas.draw()
        # self.ax.axvline(x=event.x, visible=True)

    def onrelease(self, event):
        print('%s release: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        self.clickstop = event.xdata
        allaxes = self.fig.get_axes()
        allaxes[0].axvspan(self.clickstart, self.clickstop, facecolor='green', alpha=0.5)
        self.zoomButton = Button(self.axZoom, 'Zoom', color='#90EE90')

    def ondrag(self, event):
        if event.button is None:
            return
        print('%s release: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        allaxes = self.fig.get_axes()
        allaxes[0].axvspan(self.lastDragPosition, event.xdata, facecolor='green', alpha=0.5)
        self.lastDragPosition = event.xdata

    # def zoomButton(self, event):
    # clear canvas
    # redraw canvas with xlim(onclick, onrelease)

    # def cancelButton(self, event):
    # close current fig

    def loadGroundTruthArray(self, gt_filename, start_of_analysis_sec, end_of_analysis_sec):
        gt_reader = GroundTruthReader(ConfigVAD.NO_OF_SECONDS)
        gt_array = gt_reader.load_ground_truth_array(gt_filename, start_of_analysis_sec, end_of_analysis_sec)
        print("Ground truth array: " + str(gt_array))
        return gt_array
        #  return gt_array[0:duration_in_seconds - 1]  # removing the last entry because VADLiteAdapter is one short.

    def categorize(self):
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
