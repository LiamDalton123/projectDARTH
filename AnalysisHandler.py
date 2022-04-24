import logging
import math
import numpy
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from GroundTruthReader import GroundTruthReader
from VADLite.ConfigVAD import *
from VADLiteAdapter import VADLiteAdapter


class AnalysisHandler:

    def __init__(self, sound_data_source):
        self.sound_data_source = sound_data_source

    def loadGroundTruthArray(self, gt_filename, start_of_analysis_sec, end_of_analysis_sec):
        gt_reader = GroundTruthReader(ConfigVAD.NO_OF_SECONDS)
        gt_array = gt_reader.load_ground_truth_array(gt_filename, start_of_analysis_sec, end_of_analysis_sec)
        print("Ground truth array: " + str(gt_array))
        return gt_array

    def display_analysis(self, start_time_sec, end_time_sec):
        start_tick = math.ceil(start_time_sec * self.sound_data_source.samplerate)
        end_tick = math.floor(end_time_sec * self.sound_data_source.samplerate)
        # zero out what was before start time (to keep clocks right), and ignore what was  after end time.
        analysis_buffer = numpy.concatenate([[0] * start_tick, self.sound_data_source.data[start_tick: end_tick]])
        logging.info("File name: " + self.sound_data_source.filepath)
        logging.info("Sample rate: " + str(self.sound_data_source.samplerate))
        logging.info("Length of data: " + str(len(self.sound_data_source.data)))

        fig, ax = plt.subplots(3, 2)
        plt.subplot(3, 2, 1)
        amplitude_times = np.arange(start_time_sec, end_time_sec, 1000 / self.sound_data_source.samplerate)

        plt.plot(amplitude_times, analysis_buffer[start_tick:end_tick:1000])
        plt.title = self.sound_data_source.filepath
        plt.xlabel("Time (s)")
        plt.xlim(start_time_sec, end_time_sec)
        plt.ylabel("Amplitude")
        fig.canvas.manager.set_window_title("Analysing...")
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.001)

        freqs, times, Sxx = signal.stft(analysis_buffer[:end_tick] / 32768, fs=self.sound_data_source.samplerate, window=np.hanning(2048),
                                        nfft=2048, nperseg=2048,
                                        noverlap=1024, return_onesided=True)

        plt.subplot(3, 2, (2, 6))
        plt.xlim(start_time_sec, end_time_sec)
        dB_data = 10 * np.log10(np.abs(Sxx) / np.max(np.abs(Sxx)))
        plt.pcolor(times, freqs, dB_data)  # Sxx as log
        plt.colorbar()

        gt_array = numpy.array(self.loadGroundTruthArray(self.sound_data_source.filepath + ".gt", start_time_sec, end_time_sec))
        logging.debug("gt_array length = {0}".format(len(gt_array)))
        plt.subplot(3, 2, 3)
        plt.bar(range(start_time_sec, end_time_sec), gt_array, 1.0, color='green')
        plt.xlim(start_time_sec, end_time_sec)
        plt.xlabel("Time (s)")
        plt.ylabel("GT Speech")

        vad_lite_adapter = VADLiteAdapter()
        result_array, time_array, rms_array = vad_lite_adapter.get_vad_results(analysis_buffer / 32768, self.sound_data_source.samplerate,
                                                                               start_time_sec, end_time_sec)
        # Compare gt_array and result_array
        correlation = np.corrcoef(gt_array, result_array)
        print("Pearson correlation: " + str(correlation[0, 1]))
        ax[2, 0].bar(time_array, result_array, 1.0, color='blue')
        # Twin the x-axis to make independent y-axes.
        ax[2, 0].twinx().plot(time_array, rms_array, color='red')
        plt.xlabel("Time (s)")
        plt.ylabel("Speech")
        plt.xlim(start_time_sec, end_time_sec)

        fig.canvas.manager.set_window_title(self.sound_data_source.filepath)
        plt.show(block=True)
