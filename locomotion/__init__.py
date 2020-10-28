"""
Copyright Mechanisms Underlying Behavior Lab, Singapore
https://mechunderlyingbehavior.wordpress.com/

__init__.py is part of the locomotion python package for analyzing locomotory animal 
behaviors via the techniques presented in the paper "Computational geometric tools  
for quantitative comparison of locomotory behavior" by MT Stamps, S Go, and AS Mathuru 
(https://doi.org/10.1038/s41598-019-52300-8).

This python module initializes the locomotion package when you run `import locomotion`.
"""

from locomotion.animal import *
import locomotion.trajectory as trajectory
import locomotion.heatmap as heatmap
import locomotion.write as write

__version__="1.0.0"
