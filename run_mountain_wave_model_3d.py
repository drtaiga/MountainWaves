#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:18:13 2017

@author: vaw
"""

import copy as cp
import matplotlib.pyplot as plt
import mountain_wave_model as mw_model
import numpy as np
import matplotlib.cm as cm
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8.5


run0_input = mw_model.input_params_3d()
run0_input.N_l=5.0e-5
run0_input.N_u=5.0e-5

run1_input = mw_model.input_params_3d()
run1_input.N_l=5.0e-4
run1_input.N_u=5.0e-4

run2_input = mw_model.input_params_3d()
run2_input.N_l=1.0e-3
run2_input.N_u=1.0e-3


run0 = mw_model.mountain_wave_3d(run0_input)
run0.run()

run1 = mw_model.mountain_wave_3d(run1_input)
run1.run()

run2 = mw_model.mountain_wave_3d(run2_input)
run2.run()


mw_model.comparison_plot(run0.XY,run0.YX,
                         run0.XZ,run0.ZX,
                         run0.YZ,run0.ZY,
                         run0.x, run0.y, run0.z,
                         run0.w_3d,run1.w_3d,run2.w_3d,
                         5.,0.,5.)
#plt.show()

    