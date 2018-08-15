from __future__ import absolute_import
import matplotlib
matplotlib.use('Agg')

import unittest
import os
import shutil
import logging

# Required to run the tests
from past.builtins import execfile
import tupak.core.utils

# Imported to ensure the examples run
import numpy as np
import inspect

tupak.core.utils.command_line_args.test = True


class Test(unittest.TestCase):
    outdir = 'outdir'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.path.pardir))

    @classmethod
    def setUpClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning(
                    "{} not removed prior to tests".format(self.outdir))

    @classmethod
    def tearDownClass(self):
        if os.path.isdir(self.outdir):
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                logging.warning(
                    "{} not removed prior to tests".format(self.outdir))

    def test_examples(self):
        """ Loop over examples to check they run """
        examples = ['examples/other_examples/linear_regression.py',
                    'examples/other_examples/linear_regression_unknown_noise.py',
                    ]
        for filename in examples:
            print("Testing {}".format(filename))
            execfile(filename)


if __name__ == '__main__':
    unittest.main()


