from utils import *

###############################################################################
# DEFAULT
###############################################################################
minIteration = [500] * len(paths)
# minIteration = [100] * len(paths)

lr = [0.1] * len(paths)

actFun = [0] * len(paths)

eucInit = [0] * len(paths)

###############################################################################
# PERSISTENCE DIAGRAMS
###############################################################################
lrPD = lr.copy()
lrPD[0] = 0.25  # starting
lrPD[1] = 0.005  # isabel
lrPD[3] = 0.005  # street
lrPD[4] = 10.0  # particularEnsemble
lrPD[5] = 0.5  # cloud
lrPD[6] = 0.0025  # astro
lrPD[7] = 0.0005  # impact
lrPD[8] = 0.025  # darkSky
lrPD[9] = 0.005  # volcanic
lrPD[10] = 0.0025  # astro3D
lrPD[11] = 0.005  # earthquake

lrPDCoef = lrPD.copy()
lrPDCoef[0] = 0.1  # starting
lrPDCoef[1] = 0.01  # isabel
lrPDCoef[5] = 25.0  # cloud

actFunPD = actFun.copy()
actFunPD[1] = 1  # isabel
actFunPD[4] = 1  # particular
actFunPD[6] = 1  # astro
actFunPD[7] = 1  # impact
actFunPD[9] = 1  # volcanic
actFunPD[10] = 1  # astro3D
actFunPD[11] = 1  # earthquake

eucInitPD = eucInit.copy()
# eucInitPD[7] = 1  # impact
eucInitPD[8] = 1  # darkSky

minIterationPD = minIteration.copy()
minIterationPD[0] = 1000  # starting
minIterationPD[4] = 1000  # particularEnsemble
minIterationPD[8] = 300  # darkSky

###############################################################################
# MERGE TREES
###############################################################################
lrMT = lr.copy()
# lrMT[0] = 0.04  # starting BEST
lrMT[0] = 0.05  # starting
lrMT[1] = 0.001  # isabel
lrMT[3] = 0.025  # street
# lrMT[4] = 0.02  # particularEnsemble BEST
lrMT[4] = 0.025  # particularEnsemble
lrMT[5] = 0.05  # cloud
lrMT[6] = 0.0025  # astro
lrMT[7] = 0.0025  # impact
lrMT[8] = 0.05  # darkSky
lrMT[9] = 0.001  # volcanic
lrMT[10] = 0.01  # astro3D
lrMT[11] = 0.0025  # earthquake

lrMTCoef = lrMT.copy()
lrMTCoef[3] = 0.001  # street
lrMTCoef[5] = 0.25  # cloud

actFunMT = actFun.copy()
actFunMT[6] = 1  # astro
actFunMT[8] = 1  # darkSky

eucInitMT = eucInit.copy()
eucInitMT[2] = 1  # sea
eucInitMT[8] = 1  # darkSky
eucInitMT[10] = 1  # astro3D

minIterationMT = minIteration.copy()
minIterationMT[0] = 1000  # starting
minIterationMT[4] = 1500  # particularEnsemble

###############################################################################
# FUNCTIONS
###############################################################################
def getDatasetParams(isPD, customCoef):
    if isPD:
        lrT = lrPD if customCoef else lrPDCoef
        actFunT = actFunPD
        eucInitT = eucInitPD
        minIterationT = minIterationPD
    else:
        lrT = lrMT if customCoef else lrMTCoef
        actFunT = actFunMT
        eucInitT = eucInitMT
        minIterationT = minIterationMT
    return lrT, actFunT, eucInitT, minIterationT
