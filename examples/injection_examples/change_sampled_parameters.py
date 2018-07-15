#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation sampling in non-standard parameters for an injected signal.

This example estimates the masses using a uniform prior in chirp mass, mass ratio and redshift.
The cosmology is according to the Planck 2015 data release.
"""
from __future__ import division, print_function
import tupak
import numpy as np


tupak.core.utils.setup_logger(log_level="info")

duration = 4.
sampling_frequency = 2048.
outdir = 'outdir'

np.random.seed(151226)

injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0, phi_12=1.7, phi_jl=0.3, iota=0.4,
    luminosity_distance=500, psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = tupak.gw.waveform_generator.BinaryBlackHole(
    sampling_frequency=sampling_frequency, duration=duration, waveform_arguments=waveform_arguments,
    non_standard_sampling_parameter_keys=['chirp_mass', 'mass_ratio', 'redshift'])

# Set up interferometers.
interferometers = tupak.gw.detector.InterferometerSet(['H1', 'L1', 'V1'])
interferometers.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration)
interferometers.inject_signal(parameters=injection_parameters, waveform_generator=waveform_generator)

# Set up prior
priors = tupak.gw.prior.BBHPriorSet()
[priors.pop(name) for name in ['mass_1', 'mass_2', 'luminosity_distance']]
priors['chirp_mass'] = tupak.prior.Uniform(name='chirp_mass', latex_label='$m_c$', minimum=13, maximum=45)
priors['mass_ratio'] = tupak.prior.Uniform(name='mass_ratio', latex_label='$q$', minimum=0.1, maximum=1)
priors['redshift'] = tupak.prior.Uniform(name='redshift', latex_label='$z$', minimum=0, maximum=0.5)
# These parameters will not be sampled
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra', 'dec', 'geocent_time', 'phase']:
    priors[key] = injection_parameters[key]

# Initialise GravitationalWaveTransient
likelihood = tupak.gw.likelihood.GravitationalWaveTransient(
    interferometers=interferometers, waveform_generator=waveform_generator)

# Run sampler
# We pass a function to generate all of the derived BBH parameters of interest.
result = tupak.core.sampler.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', injection_parameters=injection_parameters,
    label='DifferentParameters', outdir=outdir, conversion_function=tupak.gw.conversion.generate_all_bbh_parameters)
result.plot_corner()
