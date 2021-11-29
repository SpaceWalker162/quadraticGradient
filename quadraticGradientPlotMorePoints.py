import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
from quadraticGradientTools import *

def fFit(x, p, a):
    return p/x + a

wlSession = WolframLanguageSession('/usr/local/Wolfram/Mathematica/12.0/Executables/WolframKernel')

bestXInCenterOfMass = []
xCenter = np.array([3, 0, 0])
numberOfSpacecraftsOptions = range(10, 51, 2)
errors = np.zeros((len(numberOfSpacecraftsOptions), 3, 3, 3))

with open('bestXInCenterOfMass.json') as f:
    bestXInCenters = json.load(f)
for i in range(len(bestXInCenters)):
    arr = np.array(bestXInCenters[str(i)])
    bestXInCenterOfMass.append(arr - np.mean(arr, axis=0))


fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
for LDRatioSet, ax in zip([0.05, 0.01], axes):
    model = "magnetosphericField"
    initGradModel(wlSession, model=model, M=-30438, B0=60)
    gradB, grad2B = gradAndSecondGradB(wlSession, xCenter)
    errorsGrad2B = np.zeros((len(numberOfSpacecraftsOptions), 3, 3, 3))
    errorsGradB = np.zeros((len(numberOfSpacecraftsOptions), 3, 3))
    for ind, xInCenterOfMass in enumerate(bestXInCenterOfMass):
        numberOfSpacecrafts = xInCenterOfMass.shape[0]
        R = xInCenterOfMass.T @ xInCenterOfMass / numberOfSpacecrafts
        eigenSystemOfR = np.linalg.eig(R)
        permutation = np.argsort(eigenSystemOfR[0])
        timingShape = (np.sqrt(eigenSystemOfR[0])[permutation], eigenSystemOfR[1][:, permutation])
        contractions_ = LDRatioSet*10/timingShape[0][2]
        contraction = np.linalg.norm(xCenter)*contractions_
        x = xInCenterOfMass*contraction
        xGSEs = x*0.05 + xCenter
        if model == 'magnetosphericField':
            b = magnetosphericField(xGSE=xGSEs)
        G3All, G2All, b0All, LDRatio, eigenSystemOfR4, converged = multipointsCalculateGradAndGrad2(xGSEs, b, silence=True)
        print(LDRatio)
        errorGrad2B = 100 * np.abs((G3All[-1]-grad2B)/np.mean(np.abs(grad2B)))
        errorGradB = 100 * np.abs((G2All[-1]-gradB)/np.mean(np.abs(gradB)))
        density_ = -np.trace(G3All[-1], axis1=-1, axis2=-2)
        errorsGrad2B[ind] = errorGrad2B
        errorsGradB[ind] = errorGradB
        densitys[ind] = np.abs(density_)
    ##
    meanErrorGrad2B = errorsGrad2B.mean(axis=(1, 2, 3))
    meanErrorGradB = errorsGradB.mean(axis=(1, 2))
    print("mean gradB error: {}".format(np.mean(meanErrorGradB)))
    if model == 'magnetosphericField':
        popt, pcov = curve_fit(fFit, numberOfSpacecraftsOptions, meanErrorGrad2B)
    if model == 'magnetosphericField':
        ax.plot(numberOfSpacecraftsOptions, meanErrorGrad2B, color='b', label='$\sum_{{i,j,k}}e_{{ijk}}/27$')
        ax.plot(numberOfSpacecraftsOptions, fFit(numberOfSpacecraftsOptions, *popt), color='m', ls='--', label='${:.2f}/n+{:.2f}$'.format(*popt))
        ax.plot(numberOfSpacecraftsOptions, meanErrorGradB, color='r', label='$\sum_{{i,j}}e_{{ij}}/9$')
        ax.set_title('Constellation at [{},{},{}]$R_E$, L/D={:.2f}'.format(*xCenter, LDRatio))
        ax.set_xlabel('Measurement points n')
        ax.set_ylabel('Mean errors [percent]')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(frameon=False)
plt.tight_layout()
fig.savefig('../../latex work/secondGradient/figures/errorsVsNumberOfPointsGradB' + model + '.jpg')
fig.savefig('../../latex work/secondGradient/figures/errorsVsNumberOfPointsGradB' + model + '.pdf')
#pd.Series(bestXInCenterOfMass).to_json('bestXInCenterOfMass.json')
print("more points end")
