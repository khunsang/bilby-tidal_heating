from __future__ import absolute_import, division

import unittest
import numpy as np

import bilby
from bilby.core import utils


class TestFFT(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_nfft_frequencies(self):
        f = 2.1
        sampling_frequency = 10
        times = np.arange(0, 100, 1/sampling_frequency)
        tds = np.sin(2*np.pi*times * f + 0.4)
        fds, freqs = bilby.core.utils.nfft(tds, sampling_frequency)
        self.assertTrue(np.abs((f-freqs[np.argmax(np.abs(fds))])/f < 1e-15))

    def test_nfft_infft(self):
        sampling_frequency = 10
        tds = np.random.normal(0, 1, 10)
        fds, _ = bilby.core.utils.nfft(tds, sampling_frequency)
        tds2 = bilby.core.utils.infft(fds, sampling_frequency)
        self.assertTrue(np.all(np.abs((tds - tds2) / tds) < 1e-12))


class TestInferParameters(unittest.TestCase):

    def setUp(self):
        def source_function(freqs, a, b, *args, **kwargs):
            return None

        class TestClass:
            def test_method(self, a, b, *args, **kwargs):
                pass

        self.source1 = source_function
        test_obj = TestClass()
        self.source2 = test_obj.test_method

    def tearDown(self):
        del self.source1
        del self.source2

    def test_args_kwargs_handling(self):
        expected = ['a', 'b']
        actual = utils.infer_parameters_from_function(self.source1)
        self.assertListEqual(expected, actual)

    def test_self_handling(self):
        expected = ['a', 'b']
        actual = utils.infer_args_from_method(self.source2)
        self.assertListEqual(expected, actual)


class TestTimeAndFrequencyArrays(unittest.TestCase):

    def setUp(self):
        self.start_time = 1.3
        self.sampling_frequency = 10323
        self.duration = 8.5
        self.frequency_array = utils.create_frequency_series(sampling_frequency=self.sampling_frequency,
                                                             duration=self.duration)
        self.time_array = utils.create_time_series(sampling_frequency=self.sampling_frequency,
                                                   duration=self.duration,
                                                   starting_time=self.start_time)

    def tearDown(self):
        del self.start_time
        del self.sampling_frequency
        del self.duration
        del self.frequency_array
        del self.time_array

    def test_create_frequency_array(self):
        expected_time_array = np.array([2, 2.2, 2.4, 2.6, 2.8, 3.0])
        time_array = utils.create_time_series(sampling_frequency=5, duration=1, starting_time=2)
        self.assertTrue(np.allclose(expected_time_array, time_array))

    def test_create_time_array(self):
        expected_frequency_array = np.array([0, 1, 2, 3, 4])
        frequency_array = utils.create_frequency_series(sampling_frequency=10, duration=1)
        self.assertTrue(np.allclose(expected_frequency_array, frequency_array))

    def test_get_sampling_frequency_from_time_array(self):
        new_sampling_freq, _ = utils.get_sampling_frequency_and_duration_from_time_array(
            self.time_array)
        self.assertEqual(self.sampling_frequency, new_sampling_freq)

    def test_get_duration_from_time_array(self):
        _, new_duration = utils.get_sampling_frequency_and_duration_from_frequency_array(self.frequency_array)
        self.assertEqual(self.duration, new_duration)

    def test_get_start_time_from_time_array(self):
        new_start_time = self.time_array[0]
        self.assertEqual(self.start_time, new_start_time)

    def test_get_sampling_frequency_from_frequency_array(self):
        new_sampling_freq, new_duration = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array)
        self.assertEqual(self.sampling_frequency, new_sampling_freq)

    def test_get_duration_from_frequency_array(self):
        new_sampling_freq, new_duration = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array)
        self.assertEqual(self.duration, new_duration)

    def test_consistency_time_array_to_time_array(self):
        new_sampling_frequency, new_duration = utils.get_sampling_frequency_and_duration_from_time_array(self.time_array)
        new_start_time = self.time_array[0]
        new_time_array = utils.create_time_series(sampling_frequency=new_sampling_frequency,
                                                  duration=new_duration,
                                                  starting_time=new_start_time)
        self.assertTrue(np.allclose(self.time_array, new_time_array))

    def test_consistency_frequency_array_to_frequency_array(self):
        new_sampling_frequency, new_duration = utils.get_sampling_frequency_and_duration_from_frequency_array(self.frequency_array)
        new_frequency_array = utils.create_frequency_series(sampling_frequency=new_sampling_frequency,
                                                            duration=new_duration)
        self.assertTrue(np.allclose(self.frequency_array, new_frequency_array))


if __name__ == '__main__':
    unittest.main()
