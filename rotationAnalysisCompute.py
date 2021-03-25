#   File name: rotationAnalysisCompute.py
#   Requires: Python 3, WolframKernel (mathematica)
#   Purpose:
#       The purpose of rotation analysis is:
#           to test the S21 algorithm in modeled magnetospheric field and draw the graphs comparing the radius of curvature and helix angle
#       This is the first part that computes the distribution of the curvature and the torsion of magnetic field lines in X=0 plane in the magnetospheric field.
#
#   Record of revisions:
#       Date        author      Description of change
#       03/20/2021  Y. F. Zhou  Original code
#
#   Define variables:
#       b               -- Mangetic field
#       b0              -- b_0, approximation of the first order in Taylor series
#       G2              -- Linear gradient
#       G3              -- quadratic gradient

import numpy as np
import os
from secondDOfBTools import *

#%matplotlib auto

#   Activate Wolfram language Session
wlSession = WolframLanguageSession('/usr/local/Wolfram/Mathematica/12.0/Executables/WolframKernel')

#   Load distribution of 15 spacecraft
savedX15 = np.load('./distributionOfSpacecraft.npy')
s_ = savedX15.reshape((2, -1))
x15 = s_[1].reshape((15, 3))
xInCenterOfMass = (x15 - np.mean(x15, axis=0))

## y-z rotation frequency
planeDirection = 'X'
plane = 0
surfaceName = planeDirection + '=' + str(plane)
model = 'magnetosphericField'
subsolarDistance, alpha = magnetopauseSubsolarDistance()

#   Initialize theoretical field model in Wolfram language session
initGradModel(wlSession, model=model, M=-30438, B0=60, subsolarDistance=subsolarDistance)
initCurvatureAndTorsionModel(wlSession, model=model, M=-30438, B0=60, subsolarDistance=subsolarDistance)

#   Scale the distribution of the spacecraft
numberOfSpacecrafts = len(xInCenterOfMass)
R = xInCenterOfMass.T @ xInCenterOfMass / numberOfSpacecrafts
eigenSystemOfR = np.linalg.eig(R)
permutation = np.argsort(eigenSystemOfR[0])
timingShape = (np.sqrt(eigenSystemOfR[0])[permutation], eigenSystemOfR[1][:, permutation])
x = xInCenterOfMass/timingShape[0][0]*0.003

#   Define the range of constellation centers
yRange = np.arange(0.1, 15.5, 0.2)
zRange = np.arange(0.1, 15.5, 0.2)
xRange = np.arange(-3.1, 12.3, 0.2)

if planeDirection == 'Y':
    xCentersMat = np.zeros((len(zRange), len(xRange), 3))
elif planeDirection == 'X':
    xCentersMat = np.zeros((len(zRange), len(yRange), 3))
print("xCentersMat: {}".format(xCentersMat.shape))
if planeDirection == 'Y':
    for i, zOfGSE in enumerate(zRange):
        for j, xOfGSE in enumerate(xRange):
            xCentersMat[i, j] = np.array([xOfGSE, plane, zOfGSE])
elif planeDirection == 'X':
    for i, zOfGSE in enumerate(zRange):
        for j, yOfGSE in enumerate(yRange):
            xCentersMat[i, j] = np.array([plane, yOfGSE, zOfGSE])
xCenters = xCentersMat.reshape(-1, 3)

#   Compute curvature and torsion
curvatures, torsions, curvatures10Points, torsions10Points, LDRatio = calculateCurvaturesAndTorsions(xCenters, x, model=model, wlSession=wlSession, numberOfTurns=None, silence=False, subsolarDistance=subsolarDistance)

#   Save data
np.savez("rotationAnalysis"+model+surfaceName+".npz", xCentersMat=xCentersMat, LDRatio=LDRatio, curvatures=curvatures, torsions=torsions, curvatures10Points=curvatures10Points, torsions10Points=torsions10Points, xRange=xRange, yRange=yRange, zRange=zRange)
print("end")

#   To plot the result, run the second part of rotation analysis, rotationAnalysisPlot.py
