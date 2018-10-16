from __future__ import division, print_function
import bilby
import numpy as np

outdir = 'outdir'
label = 'ROQwBilby'

# load in the pieces for the linear part of the ROQ
B_matrix_linear = np.load("12D_IMRPhenomP/B_linear.npy")
B_matrix_linear = np.array(np.matrix(B_matrix_linear.T))
freq_nodes_linear = np.load("12D_IMRPhenomP/fnodes_linear.npy")

print(B_matrix_linear.shape)
# load in the pieces for the quadratic part of the ROQ
B_matrix_quadratic = np.load("12D_IMRPhenomP/B_quadratic.npy")
B_matrix_quadratic = np.array(np.matrix(B_matrix_quadratic.T))
freq_nodes_quadratic = np.load("12D_IMRPhenomP/fnodes_quadratic.npy")

np.random.seed(170801)

duration = 4.
sampling_frequency = 2048.

injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., iota=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50., minimum_frequency=20.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)


ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)


#make ROQ waveform generator
search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.roq,
    waveform_arguments=dict(frequency_nodes_linear=freq_nodes_linear,
                            frequency_nodes_quadratic=freq_nodes_quadratic))

priors = bilby.gw.prior.BBHPriorSet()
# priors['geocent_time'] = bilby.core.prior.Uniform(
#     minimum=injection_parameters['geocent_time'] - 1,
#     maximum=injection_parameters['geocent_time'] + 1,
#     name='geocent_time', latex_label='$t_c$', unit='$s$')
for key in ['a_1', 'a_2', 'psi', 'ra',
            'dec', 'geocent_time', 'phase']:
    priors[key] = injection_parameters[key]
for key in ['tilt_1', 'tilt_2', 'phi_12', 'phi_jl']:
    priors.pop(key)
priors['chip'] = 0
priors['alpha'] = 0


likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
    interferometers=ifos, waveform_generator=search_waveform_generator,
    linear_matrix = B_matrix_linear, quadratic_matrix = B_matrix_quadratic)

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

# Make a corner plot.
result.plot_corner()
