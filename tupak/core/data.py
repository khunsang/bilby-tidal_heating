import tupak.core.utils


class CoupledTimesFrequencies(object):

    def __init__(self, start_time, duration, sampling_frequency):
        self.__start_time = start_time
        self.__duration = duration
        self.__sampling_frequency = sampling_frequency
        self.__frequencies = None
        self.__times = None
        self.__start_time_updated = True
        self.__duration_updated = True
        self.__sampling_frequency_updated = True
        self.__frequencies_updated = False
        self.__times_updated = False

    @staticmethod
    def from_frequencies(frequency_series):
        result = CoupledTimesFrequencies(0, 0, 0)
        result.frequencies = frequency_series
        return result

    @staticmethod
    def from_times(time_series):
        result = CoupledTimesFrequencies(0, 0, 0)
        result.times = time_series
        return result

    @staticmethod
    def from_parameters(start_time, duration, sampling_frequency):
        if start_time is None or duration is None or sampling_frequency is None:
            return CoupledTimesFrequencies(0, 0, 0)
        return CoupledTimesFrequencies(start_time=start_time, duration=duration, sampling_frequency=sampling_frequency)

    @property
    def start_time(self):
        """ Allows one to set the start_time and automatically updates the time array.

        Returns
        -------
        float: The sampling frequency.

        """
        self.__update_start_time()
        return self.__start_time

    def __update_start_time(self):
        if self.__start_time_updated:
            pass
        elif self.__times_updated:
            self.__update_start_time_from_time_series()
        else:
            self.__start_time = 0
            self.__start_time_updated = True

    def __update_start_time_from_time_series(self):
        self.__start_time = self.__times[0]
        self.__start_time_updated = True

    @start_time.setter
    def start_time(self, start_time):
        self.__start_time = start_time
        self.__start_time_updated = True
        self.__times_updated = False

    @property
    def duration(self):
        """ Allows one to set the time duration and automatically updates the frequency and time array.

        Returns
        -------
        float: The time duration.

        """
        self.__update_duration()
        return self.__duration

    def __update_duration(self):
        if self.__duration_updated:
            pass
        elif self.__frequencies_updated:
            self.__update_duration_from_frequency_series()
        elif self.__times_updated:
            self.__update_duration_from_time_series()
        else:
            raise RuntimeError("No valid value for the duration could be derived")

    def __update_duration_from_time_series(self):
        _, self.__duration = tupak.core.utils.get_sampling_frequency_and_duration_from_time_array(self.times)
        self.__duration_updated = True

    def __update_duration_from_frequency_series(self):
        _, self.__duration = tupak.core.utils.get_sampling_frequency_and_duration_from_frequency_array(self.frequencies)
        self.__duration_updated = True

    @duration.setter
    def duration(self, duration):
        self.__duration = duration
        self.__duration_updated = True
        self.__reset_arrays()

    @property
    def sampling_frequency(self):
        """ Allows one to set the sampling frequency and automatically updates the frequency and time array.

        Returns
        -------
        float: The sampling frequency.

        """
        self.__update_sampling_frequency()
        return self.__sampling_frequency

    def __update_sampling_frequency(self):
        if self.__sampling_frequency_updated:
            pass
        elif self.__frequencies_updated:
            self.__update_sampling_frequency_from_frequency_array()
        elif self.__times_updated:
            self.__update_sampling_frequency_from_time_array()
        else:
            raise RuntimeError("No valid value for the sampling_frequency could be derived.")

    def __update_sampling_frequency_from_time_array(self):
        self.__sampling_frequency, _ = tupak.core.utils.get_sampling_frequency_and_duration_from_time_array(self.times)
        self.__sampling_frequency_updated = True

    def __update_sampling_frequency_from_frequency_array(self):
        self.__sampling_frequency, _ = tupak.core.utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequencies)
        self.__sampling_frequency_updated = True

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency):
        self.__sampling_frequency = sampling_frequency
        self.__sampling_frequency_updated = True
        self.__reset_arrays()

    @property
    def times(self):
        """ Time array for the data set. Automatically updates if start_time,
        sampling_frequency, or duration are updated.

        Returns
        -------
        array_like: The time array
        """
        self.__update_times()
        return self.__times

    def __update_times(self):
        if self.__times_updated:
            pass
        elif self.__sampling_frequency_updated and self.__duration_updated and self.__start_time_updated:
            self.__update_times_from_parameters()
        else:
            raise RuntimeError("No valid value for the time_array could be derived.")

    def __update_times_from_parameters(self):
        self.__update_parameters()
        self.__times = tupak.core.utils.create_time_series(sampling_frequency=self.sampling_frequency,
                                                           duration=self.duration,
                                                           starting_time=self.start_time)
        self.__times_updated = True

    @times.setter
    def times(self, time_series):
        self.__times = time_series
        self.__times_updated = True
        self.__frequencies_updated = False
        self.__reset_parameters()
        self.__update_parameters()
        self.__update_frequencies()

    @property
    def frequencies(self):
        """ Frequency array for the data set. Automatically updates if sampling_frequency or duration are updated.

        Returns
        -------
        array_like: The frequency array
        """
        self.__update_frequencies()
        return self.__frequencies

    def __update_frequencies(self):
        if self.__frequencies_updated:
            pass
        elif self.__sampling_frequency_updated and self.__duration_updated:
            self.__update_frequencies_from_parameters()
        else:
            raise RuntimeError("No valid value for the frequency_array could be derived.")

    def __update_frequencies_from_parameters(self):
        self.__update_parameters()
        self.__frequencies = tupak.core.utils.create_frequency_series(sampling_frequency=self.sampling_frequency,
                                                                      duration=self.duration)
        self.__frequencies_updated = True

    @frequencies.setter
    def frequencies(self, frequency_series):
        self.__frequencies = frequency_series
        self.__frequencies_updated = True
        self.__times_updated = False
        self.__reset_parameters()
        self.__start_time_updated = True
        self.__update_parameters()
        self.__update_times()

    def __reset_parameters(self):
        self.__sampling_frequency_updated = False
        self.__duration_updated = False
        self.__start_time_updated = False

    def __update_parameters(self):
        self.__update_duration()
        self.__update_sampling_frequency()
        self.__update_start_time()

    def __reset_arrays(self):
        self.__times_updated = False
        self.__frequencies_updated = False

    def __eq__(self, other):
        if self.sampling_frequency != other.sampling_frequency \
                or self.start_time != other.start_time \
                or self.duration != other.duration \
                or any(self.times != other.times) \
                or any(self.frequencies != other.frequencies):
            return False
        return True
