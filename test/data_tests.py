import unittest

import numpy as np
import mock
import tupak


class TestCoupledTimeAndFrequencySeriesFromInit(unittest.TestCase):

    def setUp(self):
        self.start_time = 4
        self.sampling_frequency = 1000
        self.duration = 2
        self.test_series = tupak.core.data.CoupledTimesFrequencies(start_time=self.start_time,
                                                                   sampling_frequency=self.sampling_frequency,
                                                                   duration=self.duration)

    def tearDown(self):
        del self.test_series
        del self.start_time
        del self.sampling_frequency
        del self.duration

    def test_start_time_from_init(self):
        self.assertEqual(self.start_time, self.test_series.start_time)

    def test_duration_from_init(self):
        self.assertEqual(self.duration, self.test_series.duration)

    def test_sampling_frequency_from_init(self):
        self.assertEqual(self.sampling_frequency, self.test_series.sampling_frequency)

    def test_time_array_correct_from_init(self):
        with mock.patch('tupak.core.utils.create_time_series') as m:
            m.return_value = np.array([1, 2, 3])
            self.assertTrue(np.array_equal(self.test_series.times,
                                           tupak.utils.create_time_series(sampling_frequency=self.sampling_frequency,
                                                                          duration=self.duration,
                                                                          starting_time=self.start_time)))

    def test_frequency_array_correct_from_init(self):
        with mock.patch('tupak.core.utils.create_frequency_series') as m:
            m.return_value = np.array([1, 2, 3])
            self.assertTrue(np.array_equal(self.test_series.frequencies,
                                       tupak.utils.create_frequency_series(sampling_frequency=self.sampling_frequency,
                                                                           duration=self.duration)))


class TestCoupledTimeAndFrequencySeriesSettingParameters(unittest.TestCase):

    def setUp(self):
        self.start_time = 4
        self.sampling_frequency = 1000
        self.duration = 2
        self.test_series = tupak.core.data.CoupledTimesFrequencies(start_time=self.start_time,
                                                                   sampling_frequency=self.sampling_frequency,
                                                                   duration=self.duration)

    def tearDown(self):
        del self.test_series
        del self.start_time
        del self.sampling_frequency
        del self.duration

    def test_change_start_time(self):
        self.test_series.start_time = 2
        expected_series = tupak.core.data.CoupledTimesFrequencies(sampling_frequency=self.sampling_frequency,
                                                                  duration=self.duration,
                                                                  start_time=2)
        self.assertEqual(expected_series, self.test_series)

    def test_change_duration(self):
        self.test_series.duration = 3
        expected_series = tupak.core.data.CoupledTimesFrequencies(sampling_frequency=self.sampling_frequency,
                                                                  duration=3,
                                                                  start_time=self.start_time)
        self.assertEqual(expected_series, self.test_series)

    def test_change_sampling_frequency(self):
        self.test_series.sampling_frequency = 500
        expected_series = tupak.core.data.CoupledTimesFrequencies(sampling_frequency=500,
                                                                  duration=self.duration,
                                                                  start_time=self.start_time)
        self.assertEqual(expected_series, self.test_series)


class TestCoupledTimeAndFrequencySeriesSettingTimeSeries(unittest.TestCase):

    def setUp(self):
        self.test_series = tupak.core.data.CoupledTimesFrequencies(start_time=4,
                                                                   sampling_frequency=1000,
                                                                   duration=2)
        self.expected_start_time = 1
        self.expected_sampling_frequency = 300
        self.expected_duration = 5
        self.expected_frequency_series = \
            tupak.utils.create_frequency_series(sampling_frequency=self.expected_sampling_frequency,
                                                duration=self.expected_duration)
        self.test_series.times = \
            tupak.utils.create_time_series(duration=self.expected_duration,
                                           sampling_frequency=self.expected_sampling_frequency,
                                           starting_time=self.expected_start_time)

    def tearDown(self):
        del self.test_series
        del self.expected_start_time
        del self.expected_sampling_frequency
        del self.expected_duration
        del self.expected_frequency_series

    def test_change_time_series_sets_start_time(self):
        self.assertAlmostEqual(self.expected_start_time, self.test_series.start_time, places=2)

    def test_change_time_series_sets_sampling_frequency(self):
        self.assertAlmostEqual(self.expected_sampling_frequency, self.test_series.sampling_frequency, places=2)

    def test_change_time_series_sets_duration(self):
        self.assertAlmostEqual(self.expected_duration, self.test_series.duration, places=2)

    def test_change_time_series_sets_frequency_array(self):
        _ = self.test_series.frequencies
        self.assertTrue(np.allclose(self.test_series.frequencies, self.expected_frequency_series, atol=0.1))


class TestCoupledTimeAndFrequencySeriesSettingFrequencySeries(unittest.TestCase):

    def setUp(self):
        self.expected_start_time = 4
        self.test_series = tupak.core.data.CoupledTimesFrequencies(start_time=self.expected_start_time,
                                                                   sampling_frequency=400,
                                                                   duration=6)
        self.expected_sampling_frequency = 1000
        self.expected_duration = 2
        self.test_series.frequencies = \
            tupak.utils.create_frequency_series(sampling_frequency=self.expected_sampling_frequency,
                                                duration=self.expected_duration)
        self.expected_time_series = \
            tupak.utils.create_time_series(sampling_frequency=self.test_series.sampling_frequency,
                                           duration=self.test_series.duration,
                                           starting_time=self.test_series.start_time)

    def tearDown(self):
        del self.test_series
        del self.expected_start_time
        del self.expected_sampling_frequency
        del self.expected_duration
        del self.expected_time_series

    def test_change_frequency_series_sets_start_time(self):
        self.assertAlmostEqual(self.expected_start_time, self.test_series.start_time)

    def test_change_frequency_series_sets_sampling_frequency(self):
        self.assertAlmostEqual(self.expected_sampling_frequency, self.test_series.sampling_frequency, delta=1)

    def test_change_frequency_series_sets_duration(self):
        self.assertAlmostEqual(self.expected_duration, self.test_series.duration)

    def test_change_frequency_series_sets_time_series(self):
        self.assertTrue(np.allclose(self.test_series.times, self.expected_time_series, atol=0.1))


class TestCoupledTimeAndFrequencySeriesFromFrequencySeries(unittest.TestCase):

    def setUp(self):
        self.sampling_frequency = 1000
        self.duration = 2
        self.frequency_series = tupak.utils.create_frequency_series(sampling_frequency=self.sampling_frequency,
                                                                    duration=self.duration)
        self.test_series = tupak.core.data.CoupledTimesFrequencies. \
            from_frequencies(frequency_series=self.frequency_series)

    def tearDown(self):
        del self.frequency_series
        del self.test_series
        del self.sampling_frequency
        del self.duration

    def test_start_time_from_frequency_series(self):
        self.assertEqual(0, self.test_series.start_time)

    def test_duration_from_frequency_series(self):
        self.assertEqual(self.duration, self.test_series.duration)

    def test_sampling_frequency_from_frequency_series(self):
        self.assertAlmostEqual(self.sampling_frequency, self.test_series.sampling_frequency, delta=1)

    def test_time_array_correct_from_frequency_series(self):

        def side_effect(sampling_frequency, starting_time, duration):
            return sampling_frequency + starting_time + duration

        def side_effect_2(value_1):
            return self.sampling_frequency, self.duration
        # Need to explicitly mock all of this away because utils is not 100% consistent
        with mock.patch('tupak.core.utils.create_time_series') as m:
            with mock.patch('tupak.core.utils.get_sampling_frequency_and_duration_from_frequency_array') as n:
                m.side_effect = side_effect
                n.side_effect = side_effect_2
                self.test_series = tupak.core.data.CoupledTimesFrequencies. \
                    from_frequencies(frequency_series=self.frequency_series)
                self.assertEqual(self.test_series.times,
                                 self.sampling_frequency + self.duration + self.test_series.start_time)

    def test_frequency_array_correct_from_frequency_series(self):
        self.assertTrue(np.array_equal(self.test_series.frequencies,
                                       tupak.utils.create_frequency_series(sampling_frequency=self.sampling_frequency,
                                                                           duration=self.duration)))


if __name__ == '__main__':
    unittest.main()
