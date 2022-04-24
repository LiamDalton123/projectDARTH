import logging
import math
import sys

import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib.widgets import Button
from scipy.io import wavfile


class SelectionHandler:

    def __init__(self, sound_data_source):
        self.soundDataSource = sound_data_source
        self.fig = None
        self.axZoom = None
        self.axCancel = None
        self.cancelButton = None
        self.axPlot = None
        self.click_start = None
        self.click_stop = None
        self.zoomButton = None
        self.lastDragPosition = None
        self.start_time_sec = None
        self.end_time_sec = None
        self.selections = []
        self.lastSelection = None

    def display_power_graph(self):
        self.start_time_sec = 0
        self.end_time_sec = len(self.soundDataSource.data) / self.soundDataSource.samplerate
        # zero out what was before start time (to keep clocks right), and ignore what was  after end time.
        logging.info("File name: " + self.soundDataSource.filepath)
        logging.info("Sample rate: " + str(self.soundDataSource.samplerate))
        logging.info("Length of data: " + str(len(self.soundDataSource.data)))

        self.fig, self.axPlot = plt.subplots()
        plt.title("Click and drag right to select a timespan")
        self.fig.canvas.manager.set_window_title(self.soundDataSource.filepath)
        self.axPlot.set_xlabel("Time (s)")
        self.axPlot.set_ylabel("Amplitude")
        plt.subplots_adjust(bottom=0.2)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        logging.info("cid=" + str(cid))
        cid = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        logging.info("cid=" + str(cid))
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.ondrag)
        logging.info("cid=" + str(cid))

        # buttons
        self.axZoom = plt.axes([0.65, 0.02, 0.1, 0.070])
        self.zoomButton = Button(self.axZoom, 'Analyse', hovercolor="0.85")
        self.axCancel = plt.axes([0.8, 0.02, 0.1, 0.070])
        self.cancelButton = Button(self.axCancel, 'Cancel', color='#90EE90', hovercolor="green")

        self.display_data()

        plt.ion()
        plt.show(block=True)

    def display_data(self):
        start_tick = math.ceil(self.start_time_sec * self.soundDataSource.samplerate)
        end_tick = math.floor(self.end_time_sec * self.soundDataSource.samplerate)
        analysis_buffer = numpy.concatenate([[0] * start_tick, self.soundDataSource.data[start_tick: end_tick]])
        amplitude_times = np.arange(self.start_time_sec, self.end_time_sec, 1000 / self.soundDataSource.samplerate)
        self.axPlot.plot(amplitude_times, analysis_buffer[start_tick:end_tick:1000])
        self.axPlot.set_xlim(self.start_time_sec, self.end_time_sec)

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.inaxes == self.axPlot:
            self.onclick_plot(event)
        elif event.inaxes == self.axZoom:
            self.onclick_zoom(event)
        elif event.inaxes == self.axCancel:
            self.onclick_cancel(event)

    def onrelease(self, event):
        print('%s release: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

        if event.inaxes == self.axPlot:
            self.onrelease_plot(event)
        elif event.inaxes == self.axZoom:
            self.onrelease_zoom(event)
        elif event.inaxes == self.axCancel:
            self.onrelease_cancel(event)

    def ondrag(self, event):
        if event.button is None:
            return
        print('%s drag: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.inaxes == self.axPlot:
            self.ondrag_plot(event)
        elif event.inaxes == self.axZoom:
            self.ondrag_zoom(event)
        elif event.inaxes == self.axCancel:
            self.ondrag_cancel(event)

    def onclick_plot(self, event):
        self.axPlot.axvline(x=event.xdata, visible=True, color="green", zorder=-100)
        self.click_start = event.xdata
        self.lastDragPosition = event.xdata
        self.fig.canvas.draw()

    def onclick_zoom(self, event):
        logging.debug("Zoom button clicked")

    def onclick_cancel(self, event):
        logging.debug("Cancel button clicked")

    def onrelease_plot(self, event):
        self.click_stop = event.xdata
        self.axPlot.axvspan(self.click_start, self.click_stop, facecolor='green', alpha=0.5)
        self.zoomButton.color = '#90EE90'
        self.zoomButton.hovercolor = 'green'
        self.lastSelection = (self.click_start, self.click_stop)
        self.selections.append(self.lastSelection)
        logging.debug("selection: " + str(self.lastSelection))
        logging.debug("selections:" + str(self.selections))
        self.fig.canvas.draw()

    def onrelease_zoom(self, event):
        logging.debug("Zoom button released")
        plt.close(self.fig)
        logging.debug("fig closed")

    def onrelease_cancel(self, event):
        logging.debug("Cancel button released")
        sys.exit("Cancelling run")

    def ondrag_plot(self, event):
        self.axPlot.axvspan(self.lastDragPosition, event.xdata, facecolor='green', alpha=0.5)
        self.lastDragPosition = event.xdata

    def ondrag_zoom(self, event):
        logging.debug("Zoom button dragged")

    def ondrag_cancel(self, event):
        logging.debug("Cancel button dragged")
