from unittest import TestCase

import numpy as np

from GroundTruthReader import GroundTruthReader


# speech_intervals = ["00:00:05.643/00:00:06.791", "00:00:08.595/00:00:09.514", "00:00:12.204/00:00:13.024", "00:00:15.715/00:00:16.535", "00:00:19.094/00:00:20.012", "00:00:22.801/00:00:23.555", "00:00:26.278/00:00:27.131", "00:00:29.723/00:00:30"]

class TestGroundTruthReader(TestCase):
    def test_calculate_ground_truth_array(self):
        # create test fixtures
        speech_intervals = ["00:00:10.000/00:00:20.000"]  # speech from 10 seconds to 20 seconds
        # calculate ground truth array from 0 seconds to 30 seconds
        start_of_analysis_sec = 0
        end_of_analysis_sec = 30
        frame_duration_sec = 1  # each element in the result represents one second
        gtr = GroundTruthReader(frame_duration_sec)
        number_of_expected_results_in_gt_array = (end_of_analysis_sec-start_of_analysis_sec)/frame_duration_sec
        expected_gt_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # run the test
        result = gtr.calculate_ground_truth_array(speech_intervals, start_of_analysis_sec, end_of_analysis_sec)

        # verify the result
        self.assertEqual(number_of_expected_results_in_gt_array, len(result))
        self.assertEqual(expected_gt_array.tolist(), result.tolist())
