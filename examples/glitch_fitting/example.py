#!/bin/python

from __future__ import division, print_function
import numpy as np
import tupak

# Set the duration and sampling frequency of the data segment that we're going to inject the signal into

time_duration = 4.
sampling_frequency = 2048.

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'test_runs'
tupak.core.utils.setup_logger(outdir=outdir, label=label)








