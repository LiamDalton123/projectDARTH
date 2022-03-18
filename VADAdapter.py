from numpy import ndarray


class VADAdapter:
    def get_vad_results(self, normalized_amplitudes: [], sample_rate: float, start_time_sec: float, end_time_sec: float) -> tuple[ndarray, ndarray, ndarray]:
        """
        Returns a VAD results array for the data in the buffer between start_time_sec and end_time_sec
        (along with a time array and a rms power array).

        The normalized_amplitudes is assumed to contain normalized data from start time 0, and filled at a sample rate
        of sample_rate data-points per second.

        :rtype: a tuple of three arrays of which
        the first is an analysis result array: each entry is a 1 if speech was detected, 0 otherwise.
        The second is a time array, specifying the analysis period (seconds) that each result refers to.
        The third is an rms power array for the analysis period
        :param normalized_amplitudes: must contain sound amplitudes as floats normalized to fit the range (0.0, 1.0)
        :param sample_rate: the number of samples per second in the normalized_amplitudes array
        :param start_time_sec: the time at which to start analysis
        :param end_time_sec:  the time at which to stop analysis
        """
        pass
