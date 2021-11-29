#   File name: rotationAnalysisPlot.py
#   Requires: Python 3
#   Purpose:
#       The purpose of rotation analysis is:
#           to test the S21 algorithm in modeled magnetospheric field and draw the graphs comparing the estimated radius of curvature and helix angle with the true (theoretical) one
#       This is the second part that draw the graphs comparing the radius of curvature and helix angle
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
import matplotlib.pyplot as plt
import matplotlib.patches
from quadraticGradientTools import *


#   Define the path for figure saving
saveFigPath = "./figures/"
if not os.path.exists(saveFigPath):
    os.makedirs(saveFigPath)

planeDirection = 'X'
plane = 0
surfaceName = planeDirection + '=' + str(plane)
addEarthPatch = True
model = 'magnetosphericField'
print(surfaceName)
print(model)

#   Load data
fileName = "rotationAnalysis"+surfaceName+".npz"
data = np.load(fileName)
xCentersMat = data['xCentersMat']
curvatures = data['curvatures']
torsions = data['torsions']
curvatures10Points = data['curvatures10Points']
torsions10Points = data['torsions10Points']
if planeDirection == 'X':
    xOrYRange = data['yRange']
elif planeDirection == 'Y':
    xOrYRange = data['xRange']
zRange = data['zRange']
print("LDRatios:")
print(data["LDRatio"])

curvaturesReshaped = curvatures.reshape(xCentersMat.shape[:2])
curvatures10PointsReshaped = curvatures10Points.reshape(xCentersMat.shape[:2])
torsionsReshaped = torsions.reshape(xCentersMat.shape[:2])
torsions10PointsReshaped = torsions10Points.reshape(xCentersMat.shape[:2])
radiusOfCurvatures = 1/curvaturesReshaped
helixAngle = np.arctan(torsionsReshaped * radiusOfCurvatures) * 180/np.pi
radiusOfCurvatures10Points = 1/curvatures10PointsReshaped
helixAngle10Points = np.arctan(torsions10PointsReshaped * radiusOfCurvatures10Points) * 180/np.pi

zDataNames = ["radiusOfCurvatures", "radiusOfCurvatures10Points", "helixAngle", "helixAngle10Points"]
r0, alpha = magnetopauseSubsolarDistance(Bz=27, Dp=3)
print("r0={}, alpha={}".format(r0, alpha))

#   Define the loaction of the magnetopause
if planeDirection == 'X':
    theta = np.linspace(0, np.pi/2, 50)
    r = np.repeat(r0*(2/(1 + np.cos(np.pi/2)))**alpha, len(theta))
    xyz = spherical2xyz(np.stack([r, theta, np.repeat(np.pi/2, len(theta))], axis=-1))
elif planeDirection == 'Y':
    znithAngle = np.linspace(0, np.pi*3/4, 50)
    r = r0*(2/(1 + np.cos(znithAngle)))**alpha
    theta = np.pi/2-znithAngle
    phi = np.pi*1/2*(1- np.sign(theta))
    xyz = spherical2xyz(np.stack([r, np.abs(theta), phi], axis=-1))

#   Plot
left = 0.07
bottom = 0.1
height = 0.4
width = 0.37
wholeHeight = 0.43
wholeWidth = 0.4

fig = plt.figure(figsize=(8, 7.4))
vmin = -90
vmax = 90
if planeDirection == 'Y':
    xlabel = "x [$R_E$]"
elif planeDirection == 'X':
    xlabel = "y [$R_E$]"
ylabel = "z [$R_E$]"
cbarLabels = ["$R_c$ [$R_E$]", "$\\beta$ [degrees]"]
cmaps = ["viridis",  "seismic"]
axSigns = [['a', 'b'], ['c', 'd']]
ax_xticks = np.arange(0, 16, 2)
for rowNumber in range(2):
    cmap = cmaps[rowNumber]
    cbarLabel = cbarLabels[rowNumber]
    for colNumber in range(2):
        axSign = axSigns[rowNumber][colNumber]
        zDataName = zDataNames[rowNumber*2 + colNumber]
        ax = fig.add_axes([left + colNumber*wholeWidth, bottom + (1-rowNumber)*wholeHeight, width, height])
        if rowNumber == 0:
            vm = 32
            levels = np.linspace(0, vm, 9)
            cont = ax.contourf(xOrYRange, zRange, eval(zDataName), levels=levels, cmap=cmap, vmin=0, vmax=vm)
            ax.set_xticks(ax_xticks)
        elif rowNumber == 1:
            cont = ax.contourf(xOrYRange, zRange, eval(zDataName), levels=8, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(ax_xticks)
        if planeDirection == 'Y':
            ax.plot(xyz[:, 0], xyz[:, 2], color='k', ls='--')
        elif planeDirection == 'X':
            ax.plot(xyz[:, 1], xyz[:, 2], color='k', ls='--')
        ax.text(0.5, 14, "Magnetopause")
        if addEarthPatch:
            patch = matplotlib.patches.Wedge((0, 0), 1, 0, 360, fc='grey')
            ax.add_patch(patch)
        ax.text(0.9, 0.92, '({})'.format(axSign), transform=ax.transAxes, color='k')
        ax.set_xlim(xOrYRange[[0, -1]])
        ax.set_ylim(zRange[[0, -1]])
        if rowNumber == 1:
            ax.set_xlabel(xlabel)
        elif rowNumber == 0:
            ax.tick_params(which='both', labelbottom=False)
        if colNumber == 0:
            ax.set_ylabel(ylabel)
        elif colNumber == 1:
            ax.tick_params(which='both', labelleft=False)
    axColorBar = fig.add_axes([left+wholeWidth*(colNumber+1), bottom + (1-rowNumber)*wholeHeight, 0.03, height])
    cbar = fig.colorbar(cont, cax=axColorBar)
    cbar.set_label(cbarLabel)
fig.savefig(saveFigPath + model + surfaceName + ".pdf")
fig.savefig(saveFigPath + model + surfaceName + ".jpg", dpi=300)
savedFigName = saveFigPath + model + surfaceName + ".pdf"
print(savedFigName)

print("end")
