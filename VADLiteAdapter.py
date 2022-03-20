import logging

import numpy as np
from numpy import ndarray

from VADAdapter import VADAdapter
from VADLite.ConfigVAD import ConfigVAD
from VADLite.VAD import VAD


class VADLiteAdapter(VADAdapter):
    def get_vad_result(self, buffer: [], sample_rate: int) -> tuple[int, float]:
        return VAD.classifyFrame(buffer, window_size=ConfigVAD.NO_OF_SECONDS * sample_rate), ConfigVAD.NO_OF_SECONDS

    def get_vad_results(self, normalized_amplitudes: [], sample_rate: float, start_time_sec: float, end_time_sec: float) -> tuple[ndarray, ndarray, ndarray]:
        # The results will be an array of integers, each 0 or 1, representing speech absent or present respectively
        # The length of the results array will be (end_time_sec - start_time_sec) / ConfigVAD.NO_OF_SECONDS
        result_array = np.array([])
        time_array = np.array([])
        rms_array = np.array([])

        # We will loop through every analysis period (ConfigVAD.NO_OF_SECONDS) until we reach the end time,
        # producing one analysis result for every analysis period.
        # skip over the data to the point where analysis should begin
        start_range = start_time_sec * sample_rate;
        # we will produce one analysis result per ConfigVAD.NO_OF_SECONDS
        end_range = start_range + ConfigVAD.NO_OF_SECONDS * sample_rate
        while end_range < end_time_sec * sample_rate:
            buffer = normalized_amplitudes[start_range:end_range]
            result = VAD.classifyFrame(buffer, window_size=ConfigVAD.NO_OF_SECONDS * sample_rate)
            logging.info("result = " + str(result) + ". " + ConfigVAD.PREDICTION[result])
            # Map VAD.classifyFrame result from 0=noise, 1=speech, 2 = silence to 0=non-speech, 1=speech
            if result == 2:
                result = 0
            result_array = np.append(result_array, result)
            time_array = np.append(time_array, start_range / (ConfigVAD.NO_OF_SECONDS * sample_rate))
            rms_array = np.append(rms_array, np.mean(np.sqrt(buffer ** 2)))
            start_range += ConfigVAD.NO_OF_SECONDS * sample_rate
            end_range += ConfigVAD.NO_OF_SECONDS * sample_rate

        return result_array, time_array, rms_array

