from __future__ import absolute_import, division

import unittest
import numpy as np

import bilby
from bilby.core import utils


class TestFFT(unittest.TestCase):

    def setUp(self):
        self.sampling_frequency = 10

    def tearDown(self):
        del self.sampling_frequency

    def test_nfft_sine_function(self):
        injected_frequency = 2.7324
        duration = 100
        times = utils.create_time_series(self.sampling_frequency, duration)

        time_domain_strain = np.sin(2 * np.pi * times * injected_frequency + 0.4)

        frequency_domain_strain, frequencies = bilby.core.utils.nfft(time_domain_strain, self.sampling_frequency)
        frequency_at_peak = frequencies[np.argmax(np.abs(frequency_domain_strain))]
        self.assertAlmostEqual(injected_frequency, frequency_at_peak, places=1)

    def test_nfft_infft(self):
        time_domain_strain = np.random.normal(0, 1, 10)
        frequency_domain_strain, _ = bilby.core.utils.nfft(time_domain_strain, self.sampling_frequency)
        new_time_domain_strain = bilby.core.utils.infft(frequency_domain_strain, self.sampling_frequency)
        self.assertTrue(np.allclose(time_domain_strain, new_time_domain_strain))


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
        self.sampling_frequency = 5
        self.duration = 1.6
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

    def test_create_time_array(self):
        expected_time_array = np.array([1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9])
        time_array = utils.create_time_series(sampling_frequency=self.sampling_frequency,
                                              duration=self.duration, starting_time=self.start_time)
        self.assertTrue(np.allclose(expected_time_array, time_array))

    def test_create_frequency_array(self):
        expected_frequency_array = np.array([0.0, 0.625, 1.25, 1.875, 2.5])
        frequency_array = utils.create_frequency_series(sampling_frequency=self.sampling_frequency,
                                                        duration=self.duration)
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
        new_sampling_freq, _ = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array)
        self.assertEqual(self.sampling_frequency, new_sampling_freq)

    def test_get_duration_from_frequency_array(self):
        _, new_duration = utils.get_sampling_frequency_and_duration_from_frequency_array(
            self.frequency_array)
        self.assertEqual(self.duration, new_duration)

    def test_consistency_time_array_to_time_array(self):
        new_sampling_frequency, new_duration = \
            utils.get_sampling_frequency_and_duration_from_time_array(self.time_array)
        new_start_time = self.time_array[0]
        new_time_array = utils.create_time_series(sampling_frequency=new_sampling_frequency,
                                                  duration=new_duration,
                                                  starting_time=new_start_time)
        self.assertTrue(np.allclose(self.time_array, new_time_array))

    def test_consistency_frequency_array_to_frequency_array(self):
        new_sampling_frequency, new_duration = utils.get_sampling_frequency_and_duration_from_frequency_array(self.frequency_array)
        new_frequency_array = \
            utils.create_frequency_series(sampling_frequency=new_sampling_frequency,
                                          duration=new_duration)
        self.assertTrue(np.allclose(self.frequency_array, new_frequency_array))


if __name__ == '__main__':
    unittest.main()
