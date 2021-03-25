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
#       G3              -- Quadratic gradient
#       xInCenterOfMass -- Coordinates of spacecraft in the center-of-mass system
#       xCenter         -- The center of mass of a constellation
#       LDRatio         -- L/D
#       R4              -- The fourth-order tensor R
#       R3              -- The third-order tensor R
#       RReconstructed  -- Th characteristic 6*6 tensor $\Re$ (see latex)

import numpy as np
from cycler import cycler
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
from itertools import combinations
from pprint import pprint
from scipy.signal import butter, lfilter, freqz
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import scipy.special
from wolframclient.language import wl
from wolframclient.evaluation import WolframLanguageSession

##  
def xyz2spherical(unitVectors, halfSphere=True, twopi=True):
    '''
    Transform unitVectors from Cartesian to spherical coordinates
    twopi means that phi varies from zero to 2pi
    '''
    if len(unitVectors.shape) == 1:
        unitVectors = unitVectors[None, :]
    sign0 = np.sign(unitVectors[..., 0])[..., None]
    unitVectors = unitVectors*sign0
    thetas = np.arccos(unitVectors[..., 2])
    phis = np.arcsin(unitVectors[..., 1]/np.sin(thetas))
    if not halfSphere:
        sign0 = sign0.squeeze()
        thetas = sign0*thetas + np.pi/2*(1-sign0)
        signPhis = np.sign(phis)
        phis = (1+sign0)/2*(phis+(1-signPhis)*np.pi) + (1-sign0)/2*(-phis+np.pi+np.pi/2*signPhis)
        if not twopi:
            sign1 = np.sign(np.pi-phis)
            phis -= (1-sign1)*np.pi
    return np.stack([thetas, phis], axis=-1).squeeze()

##
def spherical2xyz(vectors):
    'Transform vectors from spherical to Cartesian coordinates'
    shape = vectors.shape
    if len(shape) == 1:
        vectors = vectors[None, :]
    if shape[-1] == 2:
        x = np.sin(vectors[..., 0])*np.cos(vectors[..., 1])
        y = np.sin(vectors[..., 0])*np.sin(vectors[..., 1])
        z = np.cos(vectors[..., 0])
    elif shape[-1] == 3:
        x = vectors[..., 0]*np.sin(vectors[..., 1])*np.cos(vectors[..., 2])
        y = vectors[..., 0]*np.sin(vectors[..., 1])*np.sin(vectors[..., 2])
        z = vectors[..., 0]*np.cos(vectors[..., 1])
    return np.stack([x, y, z], axis=-1).squeeze()

##
def normalized(array, axis=-1):
    'Normalize a vector or a list of vectors'
    norm_ = np.linalg.norm(array, axis=axis)
    dim = len(array.shape)
    if dim == 1:
        array = array / norm_
    elif dim == 2:
        array = array / norm_[:, None]
    return  array


def computeGradBs(bVectorLists, xGSEs, order=1):
    '''see doi:10.1029/2002JA009612 Appendix B.
    bVectorList and xGSE in the form of [time index, spacecraft index, cartesian index]'''
    numberOfSpacecrafts = xGSEs.shape[1]
    x = xGSEs - np.mean(xGSEs, axis=1)[:, None, :]
    R = np.transpose(x, (0, 2, 1)) @ x / numberOfSpacecrafts
    RInverse = np.linalg.inv(R)
    G0 = np.transpose(bVectorLists, (0, 2, 1)) @ x @ RInverse / numberOfSpacecrafts
    if order == 1:
        LagrangianMultiplier = -np.trace(G0, axis1=1, axis2=2)/np.trace(RInverse, axis1=1, axis2=2)
        G = G0 + LagrangianMultiplier[:, None, None] * RInverse
    else:
        G = G0
    return G


def mca(bVectorLists=None, xGSEs=None, gradBs=None, bVectorAtCenters=None):
    '''see doi:10.1029/2002JA009612 equation (1)'''
    if gradBs is None:
        gradBs = computeGradBs(bVectorLists, xGSEs)
    if bVectorAtCenters is None:
        bVectorAtCenters = np.mean(bVectorLists, axis=1)
    bMagAtCenters = np.linalg.norm(bVectorAtCenters, axis=-1)
    term1 = np.sum(bVectorAtCenters[:, None, :] * gradBs, axis=2) / bMagAtCenters[:, None]**2
    term2 = - np.sum(np.sum(bVectorAtCenters[:, None, :] * gradBs, axis=2)* bVectorAtCenters, axis=1)[:, None] * bVectorAtCenters / bMagAtCenters[:, None]**4
    curvatureVectors = term1 + term2
    curvatures = np.linalg.norm(curvatureVectors, axis=1)
    normals = curvatureVectors / curvatures[:, None]
    return curvatures, normals


def magnetopauseSubsolarDistance(Bz=17, Dp=3):
    '''compute subsolar distance of the magnetopause by the model given by Shue et al, 1998 doi.org/10.1029/98JA01103'''
    r0 = (10.22 + 1.29*np.tanh(0.184*(Bz + 8.14)))*Dp**(-1/6.6)
    alpha = (0.58 - 0.007*Bz)*(1 + 0.024*np.log(Dp))
    return r0, alpha


def magnetopause(theta, subsolarDistance=None, alpha=None):
    '''Shue et al, 1998 doi.org/10.1029/98JA01103'''
    r = subsolarDistance*(2/(1 + np.cos(np.pi/2)))**alpha
    return r


def dipoleField(xGSE, M=-30438):
    '''
    return magnetic dipole field at positions defined in xGSE in RE
    B in nT
    '''
    x1 = xGSE[..., 0][..., None]
    x2 = xGSE[..., 1][..., None]
    x3 = xGSE[..., 2][..., None]
    r = np.linalg.norm(xGSE, axis=-1)[..., None]
    return M*np.concatenate([3*x1*x3, 3*x2*x3, (3*x3**2-r**2)], axis=-1) / r**5


def magnetosphericField(xGSE, M1=-30438, model="mirror", M2=-28*30438, subsolarDistance=None):
    '''
    return magnetospheric field at positions defined in xGSE in RE
    B in nT
    subsolarDistance should be in RE
    model of the magnetospheric field can be mirror, and Legendre
    '''
    if model == "mirror":
        x1 = xGSE[..., 0][..., None]
        x2 = xGSE[..., 1][..., None]
        x3 = xGSE[..., 2][..., None]
        x4 = xGSE[..., 0][..., None] - 40
        r1 = np.linalg.norm(xGSE, axis=-1)[..., None]
        xGSE2 = xGSE.copy()
        xGSE2[..., 0] = x4.squeeze()
        r2 = np.linalg.norm(xGSE2, axis=-1)[..., None]
        b1 = M1*np.concatenate([3*x1*x3, 3*x2*x3, (3*x3**2-r1**2)], axis=-1) / r1**5
        b2 = M2*np.concatenate([3*x4*x3, 3*x2*x3, (3*x3**2-r2**2)], axis=-1) / r2**5
        return b1+b2
    elif model == "Legendre":
        B1 = 2500
        B2 = 2100
        x1 = xGSE[..., 0][..., None]
        x2 = xGSE[..., 1][..., None]
        x3 = xGSE[..., 2][..., None]
        r1 = np.linalg.norm(xGSE, axis=-1)[..., None]
        b1 = M1*np.concatenate([3*x1*x3, 3*x2*x3, (3*x3**2-r1**2)], axis=-1) / r1**5
        b2 = (1/subsolarDistance)**3 * np.concatenate([-B2/subsolarDistance*x3, np.zeros_like(x3), B1-B2/subsolarDistance*x1], axis=-1)
        return b1+b2

##
def lundquistForceFreeField(x=None, R=None, B0=1, coordinateSystem='Cartesian'):
    '''
    return lundquist force free field at positions defined in x
    '''
    if not (x is None):
        r = np.linalg.norm(x[..., 0:2], axis=-1)
        theta = np.arctan(x[..., 1]/x[..., 0])
        theta = (np.sign(x[..., 0]) + 1)/2*theta - (1 - np.sign(x[..., 0]))/2*np.pi*np.sign(x[..., 1])
    bTheta = B0*scipy.special.j1(r)
    bZ = B0*scipy.special.j0(r)
    if coordinateSystem =='Cartesian':
        b = np.stack([-bTheta*np.sin(theta), bTheta*np.cos(theta), bZ], axis=-1)
    return b
##

def harrisBField(xGSE, B0=1, h=1):
    x1 = xGSE[..., 0][..., None]
    x2 = xGSE[..., 1][..., None]
    x3 = xGSE[..., 2][..., None]
    return np.concatenate([B0*np.tanh(x3/h), np.zeros_like(x3), np.zeros_like(x3)], axis=-1)


def initGradModel(session, model='dipole', M=-30438, B0=1, M1=-30438, M2=-30438*28, subsolarDistance=None, a=1, rho=1, epsilon=1):
    '''
    initialize a magnetic field model and derive the theoretical expression of its linear and quadratic gradients in Wolfram language session
    
    Define variables:
        b               -- Magnetic field
        gradB           -- Linear gradient of a magnetic field
        grad2B          -- Quadratic gradient of a magnetic field

    in dipole model if input xGSE is in RE and M=-30438, b is in nT
    in magnetosphericFieldLegendre, subsolarDistance should be in RE
    '''
    if model == 'dipole':
        initModelCMD1 = '''
        b={{3 x z M r^(-5), 3 y z M r^(-5), (3z^2-r^2) M r^(-5)}}/.{{M->{}, r->Sqrt[x^2 + y^2 + z^2]}};'''.format(M)
    elif model == 'magnetosphericField':
        initModelCMD1 = '''
        b1={{3 x z M r^(-5), 3 y z M r^(-5), (3z^2-r^2) M r^(-5)}}/.{{M->{}, r->Sqrt[x^2 + y^2 + z^2]}};
        b2={{3 (x-40) z M r^(-5), 3 y z M r^(-5), (3z^2-r^2) M r^(-5)}}/.{{M->{}, r->Sqrt[(x-40)^2 + y^2 + z^2]}};
        b=b1 + b2'''.format(M1, M2)
    elif model == 'magnetosphericFieldLegendre':
        initModelCMD1 = '''
        b1={{3 x z M r^(-5), 3 y z M r^(-5), (3z^2-r^2) M r^(-5)}}/.{{M->{}, r->Sqrt[x^2 + y^2 + z^2]}};
        b2 = (1/subsolarDistance)^3 {{-B2 z/subsolarDistance, 0, B1 - B2 x/subsolarDistance}} /. {{B1 -> 2500, B2 -> 2100}}/. {{subsolarDistance -> {}}};
        b=b1 + b2'''.format(M1, subsolarDistance)
    elif model == 'lundquistForceFreeField':
        initModelCMD1 = '''
        bCylin = {{0, B0 BesselJ[1, r], B0 BesselJ[0, r]}}/.B0->{};
        b = TransformedField["Polar" -> "Cartesian", bCylin[[{{1, 2}}]], {{r, \[Theta]}} -> {{x, y}}];
        AppendTo[b, bCylin[[3]] /. r -> Sqrt[x^2 + y^2]]; '''.format(B0)
    elif model == 'chargedSpherePotential':
        initModelCMD1 = '''b = outer (rho a^3)/(3 r epsilon) + inner ((-rho r^2)/(6 epsilon ) + (a^2 rho)/(2 epsilon)) /. {{a -> {}, rho -> {}, epsilon -> {}, r -> Sqrt[x^2 + y^2 + z^2]}};
        aa = {}'''.format(a, rho, epsilon, a)
    else:
        raise Exception("didn't find model'")
    initModelCMD2 = '''
        gradB=Grad[b, {x, y, z}] /. {{outer -> (Sign[r-aa]+1)/2, inner -> (Sign[aa-r]+1)/2}} /. r -> Sqrt[x^2 + y^2 + z^2];
        grad2B=Grad[Grad[b, {x, y, z}], {x, y, z}] /. {{outer -> (Sign[r-aa]+1)/2, inner -> (Sign[aa-r]+1)/2}} /. r -> Sqrt[x^2 + y^2 + z^2];'''
    session.evaluate(initModelCMD1)
    session.evaluate(initModelCMD2)


def gradAndSecondGradB(session, xGSE):
    '''
    Return the theoretical linear and quadratic gradients of a magnetic field model at the position defined in xGSE
    '''
    session.evaluate('''xx={};yy={};zz={};'''.format(*xGSE))
    gradBCMD = '''gradB /. {x -> xx, y -> yy, z -> zz} // N'''
    grad2BCMD = '''grad2B /. {x -> xx, y -> yy, z -> zz} // N'''
    gradB = np.array(session.evaluate(gradBCMD)).squeeze()
    grad2B = np.array(session.evaluate(grad2BCMD)).squeeze()
    return gradB, grad2B
##

def initCurvatureAndTorsionModel(session, model='dipole', M=-30438, B0=1, M1=-30438, M2=-30438*28, subsolarDistance=None):
    '''
    Initialize a magnetic field model and derive the theoretical form of the curvature and torsion of magnetic field lines
    quadratic gradient of B, get B in nT if (x,y,z) in RE
    '''
    if model == 'dipole':
        initModelCMD1 = '''b={{3 x z M r^(-5), 3 y z M r^(-5), (3z^2-r^2) M r^(-5)}}/.{{M->{}, r->Sqrt[x^2 + y^2 + z^2]}};'''.format(M)
    elif model == 'magnetosphericField':
        initModelCMD1 = '''
        b1={{3 x z M r^(-5), 3 y z M r^(-5), (3z^2-r^2) M r^(-5)}}/.{{M->{}, r->Sqrt[x^2 + y^2 + z^2]}};
        b2={{3 (x-40) z M r^(-5), 3 y z M r^(-5), (3z^2-r^2) M r^(-5)}}/.{{M->{}, r->Sqrt[(x-40)^2 + y^2 + z^2]}};
        b=b1 + b2'''.format(M1, M2)
    elif model == 'magnetosphericFieldLegendre':
        initModelCMD1 = '''
        b1={{3 x z M r^(-5), 3 y z M r^(-5), (3z^2-r^2) M r^(-5)}}/.{{M->{}, r->Sqrt[x^2 + y^2 + z^2]}};
        b2 = (1/subsolarDistance)^3 {{-B2 z/subsolarDistance, 0, B1 - B2 x/subsolarDistance}} /. {{B1 -> 2500, B2 -> 2100}}/. {{subsolarDistance -> {}}};
        b=b1 + b2'''.format(M1, subsolarDistance)
    elif model == 'lundquistForceFreeField':
        initModelCMD1 = '''
        bCylin = {{0, B0 BesselJ[1, r], B0 BesselJ[0, r]}}/.B0->{};
        b = TransformedField["Polar" -> "Cartesian", bCylin[[{{1, 2}}]], {{r, \[Theta]}} -> {{x, y}}];
        AppendTo[b, bCylin[[3]] /. r -> Sqrt[x^2 + y^2]]; '''.format(B0)
    if model in ['magnetosphericField', "magnetosphericFieldLegendre"]:
        initModelCMD2 = '''bNormalized = b/Sqrt[b.b];
        curvature = Grad[bNormalized, {x, y, z}].bNormalized;
        curvatureNorm = Sqrt[curvature.curvature];
        binormal = Cross[bNormalized, curvature/curvatureNorm];
        torsion = 1/curvatureNorm Grad[curvature, {x, y, z}].bNormalized.binormal;'''
    else:
        initModelCMD2 = '''
        bNormalized = FullSimplify[Normalize[b], {x, y, z, B0} \[Element] Reals];
        curvature = Grad[bNormalized, {x, y, z}].bNormalized;
        binormal = FullSimplify[ Normalize[Cross[b, curvature]], {x, y, z, B0} \[Element] Reals];
        torsion = FullSimplify[ 1/Norm[curvature] Grad[ curvature, {x, y, z}].bNormalized.binormal, {x, y, z, B0} \[Element] Reals]'''
    session.evaluate(initModelCMD1)
    session.evaluate(initModelCMD2)


def curvatureAndTorsion(session, xGSE):
    '''
    Return the theoretical curvature and torsion of the magnetic field line at the position defined in xGSE
    '''
    session.evaluate('''xx={};yy={};zz={};'''.format(*xGSE))
    curvatureCMD = '''curvature /. {x -> xx, y -> yy, z -> zz} // N'''
    torsionCMD = '''torsion /. {x -> xx, y -> yy, z -> zz} // N'''
    curvature = np.array(session.evaluate(curvatureCMD))
    torsion = np.array(session.evaluate(torsionCMD))
    return curvature, torsion
##

def vector2symmetricalMat(G3Reconstructed):
    length = int(np.sqrt(len(G3Reconstructed)*2))
    G3 = np.zeros((length, length))
    xs, ys = np.triu_indices(length)
    G3[xs, ys] = G3Reconstructed
    G3[ys, xs] = G3Reconstructed
    return G3


def symmetricalMat2vector(c):
    xs, ys = np.triu_indices(c.shape[-1])
    cReconstructed = c[xs, ys]
    return cReconstructed


def reconstruct(R4=None, c=None, sixCombs=[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
    RReconstructed = np.zeros((6, 6))
    cReconstructed = np.zeros((3, 6))
    for i, firstTwoIndices in enumerate(sixCombs):
        if firstTwoIndices[0] == firstTwoIndices[1]:
            factor_ = 1
        else:
            factor_ = 2
        if not (c is None):
            cReconstructed[:, i] = c[:, firstTwoIndices[0], firstTwoIndices[1]] * factor_
        if not (R4 is None):
            for k, secondTwoIndices in enumerate(sixCombs):
                if secondTwoIndices[0] == secondTwoIndices[1]:
                    RReconstructed[i, k] = R4[firstTwoIndices[0], firstTwoIndices[1], secondTwoIndices[0], secondTwoIndices[1]] * factor_
                else:
                    RReconstructed[i, k] = 2*R4[firstTwoIndices[0], firstTwoIndices[1], secondTwoIndices[0], secondTwoIndices[1]] * factor_
    return RReconstructed, cReconstructed

##
def iterateOnce(b, x, R, RInverse, R3, G3, eigenSolve=True, eigenSystemOfR4=None, numberOfSpacecrafts=None):
    '''
    To solve the equations, iterative method is needed
    '''
    dimensionOfField = b.shape[-1]
    b0 = np.mean(b, axis=0) - 1/2*np.sum(np.sum(R[None, ...]*G3, axis=-1), axis=-1)
    G = (1/numberOfSpacecrafts*b.T @ x - 1/2*np.sum(np.sum(R3[None, ...]*G3[:, None, ...], axis=-1), axis=-1)) @ RInverse
    c = 2*(1/numberOfSpacecrafts * np.sum(b.transpose((1, 0))[..., None, None] * (x[:, None, :] * x[:, :, None])[None, ...], axis=1) - b0[:, None, None] * R[None, ...] - np.sum(R3[None, ...] * G[:, None, None, :], axis=-1))
    _, cReconstructed = reconstruct(c=c)
    if eigenSolve:
        transformMat = normalized(eigenSystemOfR4[1]).T  # transformMat is \xi_{\mu i} where \mu is index of the new base and i index of the old base
        cReconstructedInNewBase = np.sum(transformMat.T[None, :, :] * cReconstructed[:, :, None], axis=1)
        G3ReconstructedInNewBase = cReconstructedInNewBase / eigenSystemOfR4[0][None, :]
        G3Reconstructed = np.sum(G3ReconstructedInNewBase[:, :, None] * transformMat[None, :, :], axis=1)
    else:
        G3Reconstructed = np.linalg.solve(RReconstructed[None, ...], cReconstructed)
    G3 = np.zeros_like(G3)
    for i in range(dimensionOfField):
        G3[i] = vector2symmetricalMat(G3Reconstructed[i])
    return b0, G, G3


def calculateEigenSystem(xGSEs):
    '''
    Calculate the eigensystem of the volumetric tensor
    '''
    x = xGSEs - np.mean(xGSEs, axis=0)
    numberOfSpacecrafts = x.shape[0]
    R = np.transpose(x, (1, 0)) @ x / numberOfSpacecrafts
    eigenSystemOfR = np.linalg.eig(R)
    R4 = 1/numberOfSpacecrafts * np.sum(x[:, :, None, None, None] * x[:, None, :, None, None] * x[:, None, None, :, None] * x[:, None, None, None, :], axis=0)
    RReconstructed, _ = reconstruct(R4=R4)
    eigenSystemOfR4 = np.linalg.eig(RReconstructed)
    return eigenSystemOfR, eigenSystemOfR4

##
def multipointsCalculateGradAndGrad2(xGSEs, b, numberOfTurns=None, silence=False, eigenSolve=True, allowNotConverge=True, x=None, xCenter=None):
    '''
    Calculate the linear and quadratic gradient of the field b at positions of xGSEs
    
    Define variables:
        numberOfTurns       -- Number of iterations. If it is set to None, the function would keep iterating until two consecutive results are close to each other enough.
        allowNotConverge    -- In the mode of numberOfTurns = None, if allowNotConverge is False and it iterates more than 10**6 times, the program raises exception. In most cases it converges when the number is less than 1000.
        x and xCenter       -- Additional information that is not necessary.
    '''
    converged = True
    dimensionOfField = b.shape[-1]
    if xCenter is None:
        xCenter = np.mean(xGSEs, axis=0)
    if x is None:
        x = xGSEs - xCenter[None, :]
    numberOfSpacecrafts = x.shape[0]
    R = np.transpose(x, (1, 0)) @ x / numberOfSpacecrafts
    eigenSystemOfR = np.linalg.eig(R)
    permutation = np.argsort(eigenSystemOfR[0])
    timingShape = (np.sqrt(eigenSystemOfR[0])[permutation], eigenSystemOfR[1][:, permutation])
    LDRatio = 2*timingShape[0][2]/np.linalg.norm(xCenter)
    RInverse = np.linalg.inv(R)
    R3 = 1/numberOfSpacecrafts * np.sum(x[:, :, None, None] * x[:, None, :, None] * x[:, None, None, :], axis=0)
    R4 = 1/numberOfSpacecrafts * np.sum(x[:, :, None, None, None] * x[:, None, :, None, None] * x[:, None, None, :, None] * x[:, None, None, None, :], axis=0)
    G3 = np.zeros((dimensionOfField, 3, 3))
    G3Last = np.ones((dimensionOfField, 3, 3))
    RReconstructed, _ = reconstruct(R4=R4)
    eigenSystemOfR4 = np.linalg.eig(RReconstructed)
    if numberOfTurns:
        G3All = np.zeros((numberOfTurns, dimensionOfField, 3, 3))
        G2All = np.zeros((numberOfTurns, dimensionOfField, 3))
        b0All = np.zeros((numberOfTurns, dimensionOfField))
        for turn in range(numberOfTurns):
            if not silence:
                print(turn)
            b0, G, G3 = iterateOnce(b, x, R, RInverse, R3, G3, eigenSolve=True, eigenSystemOfR4=eigenSystemOfR4, numberOfSpacecrafts=numberOfSpacecrafts)
            b0All[turn] = b0
            G2All[turn] = G
            G3All[turn] = G3
    elif numberOfTurns is None:
        turn = 0
        G3All = []
        while np.any(np.abs((G3-G3Last)/G3Last) > 0.00005):
#            if not silence:
#                print(turn)
            turn +=1
            if turn > 10**6:
                if allowNotConverge:
                    converged = False
                    print('not converge')
                    break
                else:
                    raise Exception('not converge')
            G3Last = G3
            b0, G, G3 = iterateOnce(b, x, R, RInverse, R3, G3, eigenSolve=True, eigenSystemOfR4=eigenSystemOfR4, numberOfSpacecrafts=numberOfSpacecrafts)
            G3All.append(G3)
        if not silence:
            print('L/D={}'.format(LDRatio))
            print('converge after {} steps'.format(turn))
        b0All = np.repeat(b0[None, ...], 2, axis=0)
        G2All = np.repeat(G[None, ...], 2, axis=0)
        G3All = np.stack(G3All)
    return G3All, G2All, b0All, LDRatio, eigenSystemOfR4, converged
##

def levicivita(arg):
    if len(arg) == 3:
        i = arg[0]
        j = arg[1]
        k = arg[2]
        return (-i+j)*(-i+k)*(-j+k)/2


def makeLeviCivitaTensor(order=3):
    if order == 3:
        levicivitaTensor = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    levicivitaTensor[i, j, k] = levicivita((i, j, k))
    return levicivitaTensor

##
def curvatureAndTorsionFormSecondGradient(b0, G2, G3, calculateTorsion=True):
    '''
    Compute curvature and torsion from b0, linear gradient, and quadratic gradient
    '''
    curvatures, normals = mca(gradBs=G2, bVectorAtCenters=b0)
    binormals = normalized(np.cross(normalized(b0), normals))
    returnedVariables = [curvatures, normals, binormals]
    if calculateTorsion:
        b0Mag = np.linalg.norm(b0, axis=-1)
        torsions = 1/(curvatures*b0Mag**3) * ((binormals[:, None, :] @ G2 @ G2 @ b0[:, :, None]).squeeze() + np.sum(np.sum(np.sum(b0[:, None, None, :] * G3, axis=-1) * b0[:, None, :], axis=-1) * binormals, axis=-1))
        returnedVariables.append(torsions)
    return returnedVariables

##
def errorVSLD(xCenter, xInCenterOfMass, numberOfContraction=20, model='lundquistForceFreeField', wlSession=None, getCurvatureAndTorsion=True, getDensity=False, errorDef='averaged', silence=False, numberOfTurns=None, returnG3CurTor=False, returnG3Density=False, returnG2AndError=False, contractionRange=None, noiseRelativeStdDev=0):
    '''
    Compute truncation error's variation with respect to L/D
    '''
    dimensionOfField = 3
    numberOfSpacecrafts = len(xInCenterOfMass)
    gradB, grad2B = gradAndSecondGradB(wlSession, xCenter)
    if getDensity:
        getCurvatureAndTorsion = False
        density = -np.trace(grad2B, axis1=-1, axis2=-2)
    if getCurvatureAndTorsion:
        curvatureVector, torsion = curvatureAndTorsion(wlSession, xCenter)
        curvature = np.linalg.norm(curvatureVector)
    ## error vs L/D
    G3LastTurnAllContraction = np.zeros((numberOfContraction, dimensionOfField, 3, 3))
    G2LastTurnAllContraction = np.zeros((numberOfContraction, dimensionOfField, 3))
    b0LastTurnAllContraction = np.zeros((numberOfContraction, dimensionOfField))
    R = xInCenterOfMass.T @ xInCenterOfMass / numberOfSpacecrafts
    eigenSystemOfR = np.linalg.eig(R)
    permutation = np.argsort(eigenSystemOfR[0])
    timingShape = (np.sqrt(eigenSystemOfR[0])[permutation], eigenSystemOfR[1][:, permutation])
    if model in ['dipole', 'magnetosphericField', 'chargedSpherePotential']:
        contractions_ = 10**np.linspace(-3, -0.5, numberOfContraction)/timingShape[0][2]
    elif model in ['lundquistForceFreeField']:
        contractions_ = np.linspace(0.01, 0.6, numberOfContraction)/timingShape[0][2]
        if noiseRelativeStdDev > 0:
            contractions_ = np.linspace(0.1, 0.6, numberOfContraction)/timingShape[0][2]
    if contractionRange is not None:
        contractions_ = contractionRange/timingShape[0][2]
    contractions = np.linalg.norm(xCenter)*contractions_ / 2
    LDRatios = np.zeros(numberOfContraction)
    for k, contraction in enumerate(contractions):
        if not silence:
            print(k)
        x = xInCenterOfMass*contraction
        xGSEs = x + xCenter
        if model == 'dipole':
            b = dipoleField(xGSEs, M=-30438)
        elif model == 'lundquistForceFreeField':
            b = lundquistForceFreeField(x=xGSEs, B0=60)
        elif model == 'magnetosphericField':
            b = magnetosphericField(xGSE=xGSEs)
        elif model == 'chargedSpherePotential':
            b = chargedSpherePotential(r=np.linalg.norm(xGSEs, axis=-1))[..., None]
        if noiseRelativeStdDev > 0:
            noiseRelative = np.random.normal(scale=noiseRelativeStdDev, size=b.shape)
            b = b * (1 + noiseRelative)
        G3All, G2All, b0All, LDRatios[k], eigenSystemOfR4, converged = multipointsCalculateGradAndGrad2(xGSEs, b, numberOfTurns=numberOfTurns, silence=silence, eigenSolve=True)
        G3LastTurnAllContraction[k] = G3All[-1]
        G2LastTurnAllContraction[k] = G2All[-1]
        b0LastTurnAllContraction[k] = b0All[-1]
    print('end')
    if errorDef == 'averaged':
        error = 100 * np.abs((G3LastTurnAllContraction-grad2B[None, ...])/np.mean(np.abs(grad2B)))
        G2Error = 100 * np.abs((G2LastTurnAllContraction-gradB[None, ...])/np.mean(np.abs(gradB)))
    else:
        error = 100 * np.abs(G3LastTurnAllContraction/grad2B[None, ...] - 1)
        G2Error = 100 * np.abs(G2LastTurnAllContraction/gradB[None, ...] - 1)
    returnedVariables = [error, LDRatios]
    if getDensity:
        densitys = -np.trace(G3LastTurnAllContraction, axis1=-1, axis2=-2)
        errorDensity = 100 * np.abs(densitys/density - 1)
        returnedVariables = [error, errorDensity, LDRatios]
    if getCurvatureAndTorsion:
        curvatures, normals, binormals, torsions = curvatureAndTorsionFormSecondGradient(b0LastTurnAllContraction, G2LastTurnAllContraction, G3LastTurnAllContraction, calculateTorsion=True)
        errorCurvatures = 100 * np.abs(curvatures/curvature - 1)
        errorTorsions = 100 * np.abs(torsions/torsion - 1)
        returnedVariables = [error, errorCurvatures, errorTorsions, LDRatios]
    if returnG3Density:
        returnedVariables.extend([densitys])
    if returnG3CurTor:
        returnedVariables.extend([G3LastTurnAllContraction, curvatures, torsions])
    if returnG2AndError:
        returnedVariables.extend([G2LastTurnAllContraction, G2Error])
    return returnedVariables


def estimationVSXCenters(xCenters, xInCenterOfMass, LDRatio=0.01, model='lundquistForceFreeField', wlSession=None, silence=False, numberOfTurns=None):
    dimensionOfField = 3
    numberOfXCenters = len(xCenters)
    numberOfSpacecrafts = len(xInCenterOfMass)
    G3LastTurnAllXCenters = np.zeros((numberOfXCenters, dimensionOfField, 3, 3))
    G2LastTurnAllXCenters = np.zeros((numberOfXCenters, dimensionOfField, 3))
    b0LastTurnAllXCenters = np.zeros((numberOfXCenters, dimensionOfField))
    G2AllXCenters = np.zeros((numberOfXCenters, dimensionOfField, 3))
    b0AllXCenters = np.zeros((numberOfXCenters, dimensionOfField))
    R = xInCenterOfMass.T @ xInCenterOfMass / numberOfSpacecrafts
    eigenSystemOfR = np.linalg.eig(R)
    permutation = np.argsort(eigenSystemOfR[0])
    timingShape = (np.sqrt(eigenSystemOfR[0])[permutation], eigenSystemOfR[1][:, permutation])
    contractions = LDRatio*np.linalg.norm(xCenters, axis=-1)/(2*timingShape[0][2])
    LDRatios = np.zeros(numberOfXCenters)
    for k, contraction in enumerate(contractions):
        if not silence:
            print(k)
        xCenter = xCenters[k]
        if wlSession is not None:
            gradB, grad2B = gradAndSecondGradB(wlSession, xCenter)
            G2AllXCenters[k] = gradB
        x = xInCenterOfMass*contraction
        xGSEs = x + xCenter
        if model == 'dipole':
            b = dipoleField(xGSEs, M=-30438)
        elif model == 'lundquistForceFreeField':
            b = lundquistForceFreeField(x=xGSEs, B0=60)
        elif model == 'magnetosphericField':
            b = magnetosphericField(xGSE=xGSEs)
        elif model == 'chargedSpherePotential':
            b = chargedSpherePotential(r=np.linalg.norm(xGSEs, axis=-1))[..., None]
            if wlSession is not None:
                b0AllXCenters[k] = chargedSpherePotential(np.linalg.norm(xCenter))
        G3All, G2All, b0All, LDRatios[k], eigenSystemOfR4, converged = multipointsCalculateGradAndGrad2(xGSEs, b, numberOfTurns=numberOfTurns, silence=silence, eigenSolve=True)
        G3LastTurnAllXCenters[k] = G3All[-1]
        G2LastTurnAllXCenters[k] = G2All[-1]
        b0LastTurnAllXCenters[k] = b0All[-1]
    if wlSession is not None:
        returnedVariables = [G2LastTurnAllXCenters, b0LastTurnAllXCenters, G2AllXCenters, b0AllXCenters, LDRatios]
    else:
        returnedVariables = [G2LastTurnAllXCenters, b0LastTurnAllXCenters, LDRatios]
    return returnedVariables

##
def calculateCurvaturesAndTorsions(xCenters, xInCenterOfMass, model='magnetosphericField', modelParas=None, getModelCurvatureAndTorsion=True, wlSession=None, numberOfTurns=None, silence=True, subsolarDistance=None):
    '''
    modelParas: model parameters.
        if model is t96:
            modelParas = [parmod, ps]  # see geopack.t96 or pypi.org/project/geopack/
    '''
    numberOfXCenters = len(xCenters)
    G3LastTurnAllXCenters = np.zeros((numberOfXCenters, 3, 3, 3))
    G2LastTurnAllXCenters = np.zeros((numberOfXCenters, 3, 3))
    b0LastTurnAllXCenters = np.zeros((numberOfXCenters, 3))
    curvatures = np.zeros(numberOfXCenters)
    torsions = np.zeros(numberOfXCenters)
    LDRatios = np.zeros(numberOfXCenters)
    for k in range(numberOfXCenters):
        if len(xInCenterOfMass.shape) == 2:
            x = xInCenterOfMass
        elif len(xInCenterOfMass.shape) > 2:
            x = xInCenterOfMass[k]
        xCenter = xCenters[k]
        print("xCenter:")
        print(xCenter)
        if model in ['dipole', 'lundquistForceFreeField', 'magnetosphericField', 'magnetosphericFieldLegendre']:
            if getModelCurvatureAndTorsion:
#                gradB, grad2B = gradAndSecondGradB(wlSession, xCenter)
                curvatureVector, torsion = curvatureAndTorsion(wlSession, xCenter)
                curvature = np.linalg.norm(curvatureVector)
                curvatures[k] = curvature
                torsions[k] = torsion
        xGSEs = x + xCenter
        if model == 'dipole':
            b = dipoleField(xGSEs, M=-30438)
        elif model == 'lundquistForceFreeField':
            b = lundquistForceFreeField(x=xGSEs, B0=60)
        elif model == 'magnetosphericField':
            b = magnetosphericField(xGSE=xGSEs)
        elif model == 'magnetosphericFieldLegendre':
            b = magnetosphericField(xGSE=xGSEs, model="Legendre", subsolarDistance=subsolarDistance)
        elif model == 'T96':
            oriShape = xGSEs.shape
            xGSEs_ = xGSEs.reshape((-1, 3))
            b = np.zeros_like(xGSEs_)
            for i in range(len(xGSEs_)):
                b[i] = np.array(geopack.dip(*xGSEs_[i])) + t96.t96(*modelParas, *xGSEs_[i])
            b.reshape(oriShape)
        G3All, G2All, b0All, LDRatios[k], eigenSystemOfR4, converged = multipointsCalculateGradAndGrad2(xGSEs, b, numberOfTurns=numberOfTurns, silence=True, eigenSolve=True)
        G3LastTurnAllXCenters[k] = G3All[-1]
        G2LastTurnAllXCenters[k] = G2All[-1]
        b0LastTurnAllXCenters[k] = b0All[-1]
    curvatures10Points, normals, binormals, torsions10Points = curvatureAndTorsionFormSecondGradient(b0LastTurnAllXCenters, G2LastTurnAllXCenters, G3LastTurnAllXCenters, calculateTorsion=True)
    if model in ['dipole', 'lundquistForceFreeField', 'magnetosphericField', 'magnetosphericFieldLegendre']:
        if getModelCurvatureAndTorsion:
            return curvatures, torsions, curvatures10Points, torsions10Points, LDRatios
    else:
        return curvatures10Points, torsions10Points, LDRatios

##
def myLegend(ax, linestyles, colors, pos, kappa=False, tau=False):
    bottom, left, width, height, wholeWidth = pos
    xs, ys = np.triu_indices(3)
    ax.text(left-wholeWidth-0.05, bottom+height*6, '$i=$', transform=ax.transAxes)
    for indIn6 in range(6):
        for indIn3 in range(3):
            xRange = np.linspace(left+wholeWidth*indIn3, left+width*(indIn3+1), 2)
            yRange = np.repeat(bottom+height*indIn6, 2)
            ax.plot(xRange, yRange, ls=linestyles[indIn3], color=colors[indIn6], transform=ax.transAxes)
        row = xs[indIn6]
        column = ys[indIn6]
        label_ = '$B_{{i,{},{}}}$'.format(row+1, column+1)
        ax.text(left-wholeWidth-0.05, bottom+height*indIn6, label_, transform=ax.transAxes)
    for indIn3 in range(3):
        label_ = '${}$'.format(indIn3+1)
        ax.text(left+wholeWidth*indIn3, bottom+height*6, label_, transform=ax.transAxes)
    if kappa:
        ax.text(left+wholeWidth*3, bottom+height*5, '$\delta \kappa$', transform=ax.transAxes)
        ax.plot(np.linspace(left+wholeWidth*4, left+wholeWidth*4+width, 2), np.repeat(bottom+height*5+height/4, 2), ls='-.', color='k', transform=ax.transAxes)
    if tau:
        ax.text(left+wholeWidth*3, bottom+height*4, '$\delta \\tau$', transform=ax.transAxes)
        ax.plot(np.linspace(left+wholeWidth*4, left+wholeWidth*4+width, 2), np.repeat(bottom+height*4+height/4, 2), ls=':', color='k', transform=ax.transAxes)
