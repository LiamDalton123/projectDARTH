import datetime
import logging
import string
from datetime import time

import numpy as np


class GroundTruthReader:

    def __init__(self, frame_duration_sec):
        """
        Creates a GroundTruthReader for a VAD algorithm that splits an analysis into discrete frames
        (each of frame_duration_sec seconds)
        :param frame_duration_sec: a real number indicating how long each analysis frame lasts (typically 1 sec).
        """
        self.frame_duration_sec = frame_duration_sec

    @staticmethod
    def load_ground_truth_speech_intervals(gt_filename: string) -> [string]:
        """
        Loads ground truth speech intervals from a file.
        a speech interval is in the format hh:mm:ss.sss/hh:mm:ss.sss which is an ISO8601 interval format
        For example 01:02:03.456/02:04:06.789 represents an interval beginning at 1 hour, 2 minutes and 3.456 seconds
        and ending at 2 hours 4 minutes and 6.789 seconds.

        It is expected that the start time (00:00:00.000) represents the start of the audio file for which the gt_file
        contains the ground truth.
        :param gt_filename: the name of a file containing ground truth speech intervals
        :return: an array of speech interval strings corresponding to those in the file.
        """
        speech_intervals = []
        with open(gt_filename) as file:
            lines = file.readlines()
        for line in lines:
            speech_intervals.append(line.strip())
        return speech_intervals

    def load_ground_truth_array(self, gt_filename, start_of_analysis_in_seconds, end_of_analysis_in_seconds):
        """
        Loads ground truth array for an analysis period, based on speech intervals defined in the gt file
        (see 'load_ground_truth_speech_intervals' for more info on the format of the file).
        The period of analysis is split into a number of frames (as required by the VADAdapter),
        each of frame_duration_sec, and each result in the returned ground truth indicates of speech is present
        or not in that frame.
       :param gt_filename: the name of a file containing ground truth speech intervals
        :return: an array of ground truths one per frame, indicating is speech is present (1) or not (0)
        in each frame
        """
        speech_intervals = GroundTruthReader.load_ground_truth_speech_intervals(gt_filename)
        return self.calculate_ground_truth_array(speech_intervals,
                                                 start_of_analysis_in_seconds,
                                                 end_of_analysis_in_seconds)

    def calculate_ground_truth_array(self, speech_intervals, start_of_analysis_in_seconds, end_of_analysis_in_seconds):
        duration_in_seconds = end_of_analysis_in_seconds - start_of_analysis_in_seconds
        gt_array = np.array([])
        current_time_in_seconds = start_of_analysis_in_seconds
        speech_end_time_in_seconds = 0
        for interval in speech_intervals:
            speech_start_string, speech_end_string = interval.split("/")
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
        return gt_array[0:duration_in_seconds]






