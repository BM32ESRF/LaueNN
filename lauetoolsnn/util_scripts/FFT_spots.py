# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:48:38 2022

@author: PURUSHOT



Idea : FFT to spots seperation in Laue data

Finding a batch pair of spots, it is perhaps better to apply fourier transform to the spots
and following the same pattern throughout the image; sort of tlike Image template matching

Similarly create Fourier patterns of perfect nd imperfect spots
and use at as a template for the CNN sorting algortihm
"""

