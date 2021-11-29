#   File name: quadraticGradientBPlot.py
#   Requires: Python 3, WolframKernel (mathematica)
#   Purpose:
#       To test the S21 algorithm in flux rope, dipole field, and modeled magnetospheric field and draw the graphs showing Errors-Iterations and Errors-L/D
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
#       xInCenterOfMass -- coordinates of spacecraft in the center-of-mass system
#       xCenter         -- the center of mass of a constellation
#       LDRatio         -- L/D


import numpy as np
import os
import matplotlib.pyplot as plt
from quadraticGradientTools import *

#   Activate Wolfram language session
wlSession = WolframLanguageSession('/usr/local/Wolfram/Mathematica/12.0/Executables/WolframKernel')

#   Load distribution of 15 spacecraft
savedX15 = np.load('distributionOfSpacecraft.npy')
s_ = savedX15.reshape((2, -1))
x15 = s_[1].reshape((15, 3))
xInCenterOfMass = (x15 - np.mean(x15, axis=0))

#   Define the path for figure saving
saveFigPath = './figures'
if not os.path.exists(saveFigPath):
    os.makedirs(saveFigPath)


#   Define plot styles
linestyles = ['--', ':', '-']
colors = ['y', 'b', 'r', 'c', 'g', 'm']
lw = 1

#   Define models the tests will be on
models = ['lundquistForceFreeField', 'dipole', 'magnetosphericField']

#   Test on defined models in for loop
for model in models:
    
    plotXaxis = 'iteration'  # else plot L/D
    plotLD = True

    #   Initialize theoretical field model in Wolfram language session
    initGradModel(wlSession, model=model, M=-30438, B0=60)
    initCurvatureAndTorsionModel(wlSession, model=model, M=-30438, B0=60)
    dimensionOfField = 3
    #   define constellation positions for test
    if model == 'lundquistForceFreeField':
        xCenters = np.array([[1, 0, 0], [0.5, 0, 0], [0.1, 0, 0]])
    elif model == 'dipole':
        xCenters = np.array([[3, 0, 0], [2, 0, 3], [0, 0, 3]])
    elif model == 'magnetosphericField':
        xCenters = np.array([[-5, 15, 10], [5, 10, 10], [5, 15, 5]])

    xCenter = xCenters[0]
    numberOfSpacecrafts = len(xInCenterOfMass)
    dropped = []
    if plotXaxis == 'iteration':
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))

        #   scale the distribution of the spacecraft
        R = xInCenterOfMass.T @ xInCenterOfMass / numberOfSpacecrafts
        eigenSystemOfR = np.linalg.eig(R)
        permutation = np.argsort(eigenSystemOfR[0])
        timingShape = (np.sqrt(eigenSystemOfR[0])[permutation], eigenSystemOfR[1][:, permutation])
        x = xInCenterOfMass/timingShape[0][0]*0.01*np.linalg.norm(xCenter)
        xGSEs = x + xCenter

        #   Define the components of quadratic gradient that vanishes due to the symmetry in each field model and will not be plotted
        if model == 'dipole':
            if np.all(xCenter == xCenters[0]):
                dropped = [(0,0,0),(0,0,1),(0,1,1),(0,1,2),(0,2,2),(1,0,0),(1,0,1),(1,0,2),(1,1,1),(1,2,2),(2,0,1),(2,0,2),(2,1,2)]
            elif np.all(xCenter == xCenters[1]):
                dropped = [(0,0,1),(0,1,2),(1,0,0),(1,0,2),(1,1,1),(1,2,2),(2,0,1),(2,1,2)]
            elif np.all(xCenter == xCenters[2]):
                dropped = [(0,0,0),(0,0,1),(0,1,1),(0,1,2),(0,2,2),(1,0,0),(1,0,1),(1,0,2),(1,1,1),(1,2,2),(2,0,1),(2,0,2),(2,1,2)]
        elif model == 'lundquistForceFreeField':
            dropped = [(0,0,2),(0,1,2),(0,2,2),(1,0,2),(1,1,2),(1,2,2),(2,0,2),(2,1,2),(2,2,2),(0,0,0),(0,1,1),(1,0,1),(2,0,1)]

        #   Return the field at positions defined as xGSEs
        if model == 'dipole':
            b = dipoleField(xGSEs, M=-30438)
        elif model == 'lundquistForceFreeField':
            b = lundquistForceFreeField(x=xGSEs, B0=60)
        elif model == 'magnetosphericField':
            b = magnetosphericField(xGSE=xGSEs)
        elif model == 'chargedSpherePotential':
            b = chargedSpherePotential(r=np.linalg.norm(xGSEs, axis=-1))[..., None]

        #   Iterate the algorithm 1,000 times and return the result
        G3All, G2All, b0All, LDRatio, eigenSystemOfR4, converged = multipointsCalculateGradAndGrad2(xGSEs, b, numberOfTurns=1000, silence=True)

        #   Calculate the theoretical values of the linear and quadratic gradients using Wolfram language
        gradB, grad2B = gradAndSecondGradB(wlSession, xCenter)

        #   Calculate relative errors
        if model in ['dipole', 'lundquistForceFreeField']:
            error = 100 * np.abs(G3All/grad2B[None, ...] - 1)
            G2Error = 100 * np.abs(G2All/gradB[None, ...] - 1)
        elif model in ['magnetosphericField']:
            error = 100 * np.abs((G3All-grad2B[None, ...])/np.mean(np.abs(grad2B)))
            G2Error = 100 * np.abs((G2All-gradB[None, ...])/np.mean(np.abs(gradB)))
        
        #   Plot
        columnNumber = 0
        rowNumber = 1
        if rowNumber == 1:
            xs, ys = np.triu_indices(3)
            for i in range(dimensionOfField):
                for indIn6, row, column in zip(range(6), xs, ys):
                    for dropped_ in dropped:
                        if ((i, row, column) == dropped_):
                            break
                    else:
                        if model == 'magnetosphericField':
                            axes[rowNumber].plot(np.arange(len(G3All))+1, error[:, i, row, column], label='$B_{{{},{},{}}}$'.format(i+1, row+1, column+1), ls=linestyles[i], color=colors[indIn6], lw=lw)
                        else:
                            axes[rowNumber].plot(np.arange(len(G3All))+1, error[:, i, row, column], label='$B_{{{},{},{}}}$'.format(i+1, row+1, column+1), lw=lw)
            axes[rowNumber].set_xlim([-0.5, 50])
            axes[rowNumber].set_ylim([0, 70])
        rowNumber = 0
        if columnNumber == 0:
            for i in range(dimensionOfField):
                for j in range(3):
                    if G2Error[0, i, j] < np.inf:
                        if dimensionOfField == 1:
                            axes[rowNumber].plot(np.arange(len(G2All))+1, G2Error[:, i, j], label='$\\phi_{{,{}}}$'.format(j+1), ls='-', color='k')
                        else:
                            axes[rowNumber].plot(np.arange(len(G2All))+1, G2Error[:, i, j], label='$B_{{{},{}}}$'.format(i+1,j+1), lw=lw)
            axes[rowNumber].set_xlim([-0.5, 50])
            if model in ['lundquistForceFreeField']:
                ylim_ = axes[rowNumber].get_ylim()
                axes[rowNumber].set_ylim([0, ylim_[1]])
            else:
                axes[rowNumber].set_ylim([0, 4])
        LDRatio_ = LDRatio
        gradientTypes = ['linear', 'quadratic']
        figureTokens = ['a', 'b']
        for rowNumber in range(2):
            gradientType = gradientTypes[rowNumber]
            figureToken = figureTokens[rowNumber]
            axes[rowNumber].set_title('Constellation at [{},{:.0f},{:.0f}]$R_E$'.format(*xCenter))
            axes[rowNumber].text(0.45, 0.9, '({})'.format(figureToken), transform=axes[rowNumber].transAxes, color='k')
            if model =='dipole':
                axes[rowNumber].legend(frameon=False, ncol=1)
                axes[rowNumber].text(0.75, 0.3, 'L/D={:.3f}'.format(LDRatio_), transform=axes[rowNumber].transAxes, color='k')
            elif model =='lundquistForceFreeField':
                axes[rowNumber].legend(frameon=False, ncol=1)
                axes[rowNumber].text(0.75, 0.3, 'L/D={:.3f}'.format(LDRatio_), transform=axes[rowNumber].transAxes, color='k')
            elif model == 'magnetosphericField':
                if rowNumber == 1:
                    bottom = 0.2
                    left = 0.7
                    width = 0.09
                    height = 0.1
                    wholeWidth = 0.12
                    pos = [bottom, left, width, height, wholeWidth]
                    myLegend(axes[rowNumber], linestyles, colors, pos)
                    axes[rowNumber].text(0.75, 0.9, 'L/D={:.3f}'.format(LDRatio_), transform=axes[rowNumber].transAxes, color='k')
                elif rowNumber == 0:
                    axes[rowNumber].legend(frameon=False, ncol=3, loc=(0.05, 0.5))
                    axes[rowNumber].text(0.75, 0.3, 'L/D={:.3f}'.format(LDRatio_), transform=axes[rowNumber].transAxes, color='k')
            axes[rowNumber].set_xlabel('Iterations [count]')
            axes[rowNumber].set_ylabel('Errors [percent]')
        plt.tight_layout()
        figName = plotXaxis + 'Error' + model
        fig.savefig(os.path.join(saveFigPath, '{}.pdf'.format(figName)))
        fig.savefig(os.path.join(saveFigPath, '{}.jpg'.format(figName)))

    # Plot error vs L/D
    if plotLD:
        plotXaxis = 'LD'
    if plotXaxis == 'LD':
        plotErrors = ['G2', 'G3']
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 9))
        axSigns = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        if model in ['dipole', 'lundquistForceFreeField']:
            errorDef = None
        elif model == 'magnetosphericField':
            errorDef = 'averaged'
        for xCenterInd, xCenter in enumerate(xCenters):
            #   Define the components of quadratic gradient that vanishes due to the symmetry in each field model and will not be plotted
            if model == 'dipole':
                if np.all(xCenter == xCenters[0]):
                    dropped = [(0,0,0),(0,0,1),(0,1,1),(0,1,2),(0,2,2),(1,0,0),(1,0,1),(1,0,2),(1,1,1),(1,2,2),(2,0,1),(2,0,2),(2,1,2)]
                elif np.all(xCenter == xCenters[1]):
                    dropped = [(0,0,1),(0,1,2),(1,0,0),(1,0,2),(1,1,1),(1,2,2),(2,0,1),(2,1,2)]
                elif np.all(xCenter == xCenters[2]):
                    dropped = [(0,0,0),(0,0,1),(0,1,1),(0,1,2),(0,2,2),(1,0,0),(1,0,1),(1,0,2),(1,1,1),(1,2,2),(2,0,1),(2,0,2),(2,1,2)]
            if model == 'lundquistForceFreeField':
                dropped = [(0,0,2),(0,1,2),(0,2,2),(1,0,2),(1,1,2),(1,2,2),(2,0,2),(2,1,2),(2,2,2),(0,0,0),(0,1,1),(1,0,1),(2,0,1)]
            rowNumber = xCenterInd
            for columnNumber in range(2):
                plotError = plotErrors[columnNumber]
                axSign = axSigns[xCenterInd][columnNumber]
                if plotError == 'G3':
                    error, errorCurvatures, errorTorsions, LDRatios, G3LastTurnAllContraction, curvatures, torsions = errorVSLD(xCenter, xInCenterOfMass, numberOfContraction=20, model=model, wlSession=wlSession, returnG3CurTor=True, numberOfTurns=None, silence=True, errorDef=errorDef, noiseRelativeStdDev=noiseRelativeStdDev)
                elif plotError == 'G2':
                    error, errorCurvatures, errorTorsions, LDRatios, G3LastTurnAllContraction, curvatures, torsions, G2LastTurnAllContraction, G2Error = errorVSLD(xCenter, xInCenterOfMass, numberOfContraction=20, model=model, wlSession=wlSession, returnG3CurTor=True, numberOfTurns=None, silence=True, errorDef=errorDef, returnG2AndError=True, noiseRelativeStdDev=noiseRelativeStdDev)
                if plotError == 'G3':
                    xs, ys = np.triu_indices(3)
                    labels = []
                    plots = []
                    for i in range(dimensionOfField):
                        for indIn6, row, column in zip(range(6), xs, ys):
                            for dropped_ in dropped:
                                if ((i, row, column) == dropped_):
                                    break
                            else:
                                label_ = '$B_{{{},{},{}}}$'.format(i+1, row+1, column+1)
                                if model == 'magnetosphericField':
                                    plot_, = axes[rowNumber][columnNumber].plot(LDRatios, error[:, i, row, column], label=label_, ls=linestyles[i], color=colors[indIn6], lw=lw)
                                else:
                                    plot_, = axes[rowNumber][columnNumber].plot(LDRatios, error[:, i, row, column], label=label_, lw=lw)
                                plots.append(plot_)
                                labels.append(label_)
                elif plotError == 'G2':
                    labels = []
                    plots = []
                    for i in range(dimensionOfField):
                        for j in range(3):
                            if G2Error[0, i, j] < np.inf:
                                label_ = '$B_{{{},{}}}$'.format(i+1,j+1)
                                plot_, = axes[rowNumber][columnNumber].plot(LDRatios, G2Error[:, i, j], label=label_, lw=lw)
                                plots.append(plot_)
                                labels.append(label_)
                if model == 'lundquistForceFreeField':
                    if plotError == 'G3':
                        axes[rowNumber][columnNumber].set_ylim([0, 6])
                        pass
                    elif plotError == 'G2':
                        ylim_ = axes[rowNumber][columnNumber].get_ylim()
                        axes[rowNumber][columnNumber].set_ylim([0, ylim_[1]])
                if model in ['dipole', 'magnetosphericField']:
                    axes[rowNumber][columnNumber].set_xscale('log')
                    axes[rowNumber][columnNumber].set_xlim([0.003,0.13])
                    axes[rowNumber][columnNumber].set_ylim([0,70])
                    if plotError == 'G3':
                        if xCenterInd == 1:
                            axes[rowNumber][columnNumber].set_ylim([0,30])
                        else:
                            if model == 'magnetosphericField':
                                axes[rowNumber][columnNumber].set_ylim([0,30])
                            else:
                                axes[rowNumber][columnNumber].set_ylim([0,30])
                    elif plotError == 'G2':
                        axes[rowNumber][columnNumber].set_ylim([0,10])
                axes[rowNumber][columnNumber].set_xlabel('L/D')
                axes[rowNumber][columnNumber].set_ylabel('Errors [percent]')
                axes[rowNumber][columnNumber].set_title('Constellation at [{},{:.0f},{:.0f}]$R_E$'.format(*xCenter))
                if plotError == 'G2':
                    if model == 'dipole' and xCenterInd == 2:
                        pass
                    else:
                        plot_, = axes[rowNumber][columnNumber].plot(LDRatios, errorCurvatures, label='$\delta \kappa$', linestyle='-.', color='k', lw=lw)
                if plotError == 'G3':
                    if model == 'lundquistForceFreeField' or model == 'magnetosphericField':
                        axes[rowNumber][columnNumber].plot(LDRatios, errorTorsions, label='$\delta \\tau$', linestyle=':', color='k', lw=lw)
                axes[rowNumber][columnNumber].set_xlabel('L/D')
                axes[rowNumber][columnNumber].set_ylabel('Errors [percent]')
                axes[rowNumber][columnNumber].text(0.45, 0.9, '({})'.format(axSign), transform=axes[rowNumber][columnNumber].transAxes, color='k')
                axes[rowNumber][columnNumber].set_title('Constellation at [{},{:.0f},{:.0f}]$R_E$'.format(*xCenter))
                if plotError == 'G3' and model == 'dipole' and xCenterInd == 1:
                    leg1 = axes[rowNumber][columnNumber].legend(plots[:7], labels[:7], frameon=False, loc=(0.01, 0.15))
                    leg2 = axes[rowNumber][columnNumber].legend(plots[7:], labels[7:], frameon=False, loc=(0.31, 0.35))
                    axes[rowNumber][columnNumber].add_artist(leg1)
                elif model == 'magnetosphericField':
                    if plotError == 'G2':
                        labels.append('$\delta \kappa$')
                        plots.append(plot_)
                        leg1 = axes[rowNumber][columnNumber].legend(plots[:-1], labels[:-1], frameon=False, ncol=3, loc=(0.05, 0.5))
                        leg2 = axes[rowNumber][columnNumber].legend([plots[-1]], [labels[-1]], frameon=False, loc=(0.34, 0.38))
                        axes[rowNumber][columnNumber].add_artist(leg1)
                    elif plotError == 'G3':
                        bottom = 0.3
                        left = 0.16
                        width = 0.07
                        height = 0.1
                        wholeWidth = 0.1
                        pos = [bottom, left, width, height, wholeWidth]
                        myLegend(axes[rowNumber][columnNumber], linestyles, colors, pos, tau=True)
                elif plotError == 'G3' and model == 'lundquistForceFreeField' and xCenterInd == 2:
                    axes[rowNumber][columnNumber].legend(frameon=False, ncol=1, loc='upper left')
                else:
                    axes[rowNumber][columnNumber].legend(frameon=False, ncol=1)
        plt.tight_layout()
        figName = plotXaxis + 'Error' + model
        fig.savefig(os.path.join(saveFigPath, '{}.pdf'.format(figName)))
        fig.savefig(os.path.join(saveFigPath, '{}.jpg'.format(figName)))
    print("model: {} all end".format(model))
