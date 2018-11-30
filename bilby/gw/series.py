import numpy as np

from ..core import utils


class CoupledTimeAndFrequencySeries(object):

    def __init__(self, duration=None, sampling_frequency=None, start_time=0):
        """ A waveform generator

    Parameters
    ----------
    sampling_frequency: float, optional
        The sampling frequency
    duration: float, optional
        Time duration of data
    start_time: float, optional
        Starting time of the time array
        """
        self._duration = duration
        self._sampling_frequency = sampling_frequency
        self._check_legal_duration_and_sampling_frequency()
        self.start_time = start_time
        self._frequency_array_updated = False
        self._time_array_updated = False
        self._frequency_array = None
        self._time_array = None

    def __repr__(self):
        return self.__class__.__name__ + '(duration={}, sampling_frequency={}, start_time={})'\
            .format(self.duration, self.sampling_frequency, self.start_time)

    @property
    def frequency_array(self):
        """ Frequency array for the waveforms. Automatically updates if sampling_frequency or duration are updated.

        Returns
        -------
        array_like: The frequency array
        """
        if self._frequency_array_updated is False:
            if self.sampling_frequency and self.duration:
                self._frequency_array = utils.create_frequency_series(
                    sampling_frequency=self.sampling_frequency,
                    duration=self.duration)
            else:
                raise ValueError('Can not calculate a frequency series without a '
                                 'legitimate sampling_frequency ({}) or duration ({})'
                                 .format(self.sampling_frequency, self.duration))

        return self._frequency_array

    @frequency_array.setter
    def frequency_array(self, frequency_array):
        self._frequency_array = frequency_array
        self._sampling_frequency, self._duration = \
            utils.get_sampling_frequency_and_duration_from_frequency_array(frequency_array)
        self._frequency_array_updated = True

    @property
    def time_array(self):
        """ Time array for the waveforms. Automatically updates if sampling_frequency or duration are updated.

        Returns
        -------
        array_like: The time array
        """

        if self._time_array_updated is False:
            if self.sampling_frequency and self.duration:
                self._time_array = utils.create_time_series(
                    sampling_frequency=self.sampling_frequency,
                    duration=self.duration,
                    starting_time=self.start_time)
            else:
                raise ValueError('Can not calculate a time series without a '
                                 'legitimate sampling_frequency ({}) or duration ({})'
                                 .format(self.sampling_frequency, self.duration))

            self._time_array_updated = True
        return self._time_array

    @time_array.setter
    def time_array(self, time_array):
        self._time_array = time_array
        self._sampling_frequency, self._duration = \
            utils.get_sampling_frequency_and_duration_from_time_array(time_array)
        self._start_time = time_array[0]
        self._time_array_updated = True

    @property
    def duration(self):
        """ Allows one to set the time duration and automatically updates the frequency and time array.

        Returns
        -------
        float: The time duration.

        """
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration
        self._check_legal_duration_and_sampling_frequency()
        self._frequency_array_updated = False
        self._time_array_updated = False

    @property
    def sampling_frequency(self):
        """ Allows one to set the sampling frequency and automatically updates the frequency and time array.

        Returns
        -------
        float: The sampling frequency.

        """
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency):
        self._sampling_frequency = sampling_frequency
        self._check_legal_duration_and_sampling_frequency()
        self._frequency_array_updated = False
        self._time_array_updated = False

    def _check_legal_duration_and_sampling_frequency(self):
        if self._sampling_frequency is None or self._duration is None:
            return
        num = self._sampling_frequency * self._duration
        tol = 1e-8
        if np.abs(num - np.round(num)) > tol:
            raise IllegalDurationAndSamplingFrequencyException(
                '\nYour sampling frequency and duration must multiply to a number'
                'close to (tol = {}) an integer number. \nBut sampling_frequency={} and '
                'duration={} multiply to {}'.format(
                    tol, self._sampling_frequency, self._duration,
                    self._sampling_frequency*self._duration
                )
            )
        self._duration = np.round(self._sampling_frequency * self._duration) / self.sampling_frequency

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = start_time
        self._time_array_updated = False


class IllegalDurationAndSamplingFrequencyException(Exception):
    pass
