import argparse
import multiprocessing
import re
from utils import *
from hyperParameters import *


def getParamsCorr():
    corr = {
        # Input Options
        "-d": ["dirPath", "", str],
        "-dl": ["debugLevel", 3, int],
        "-nt": ["noThreads", multiprocessing.cpu_count(), int],
        "-pd": ["isPD", 0, int],
        "-dt": ["deterministic", 1, int],
        "-c": ["coef", 0.0, float],
        "-e1": ["eps1", 5.0, float],
        "-e2": ["eps2", 95.0, float],
        "-e3": ["eps3", 90.0, float],
        "-pt": ["pt", 1.0, float],
        "-ptMult": ["ptMult", 1.0, float],
        "-pga": ["pga", 0, int],
        # Architecture Options
        "-nl": ["noLayers", 0, int],
        "-iNG": ["inputNoGeodesics", 16, int],
        "-iBsl": ["inputBarySizeLimit", 15.0, float],
        "-lNG": ["latentNoGeodesics", 2, int],
        "-lBsl": ["latentBarySizeLimit", 10.0, float],
        "-bsl": ["barySizeLimit", 20.0, float],
        "-a": ["activate", 1, int],
        "-af": ["activationFunction", 0, int],
        "-fs": ["fullSymmetricAE", 0, int],
        "-aoi": ["activateOutputInit", 0, int],
        # Optimization Options
        "-mi": ["minIteration", 500, int],
        "-mai": ["maxIteration", 0, int],
        "-ig": ["iterationGap", 100, int],
        "-bs": ["batchSize", 1.0, float],
        "-op": ["optimizer", 0, int],
        "-lr": ["learningRate", 0.1, float],
        "-b1": ["beta1", 0.9, float],
        "-b2": ["beta2", 0.999, float],
        "-rw": ["reconstructionLossWeight", 1.0, float],
        "-mw": ["metricLossWeight", 0.0, float],
        "-cw": ["clusteringLossWeight", 0.0, float],
        "-cwt": ["clusteringLossTemperature", 5.0, float],
        "-dw": ["customLossDynamicWeight", 0, int],
        "-cls": ["customLossSpace", 0, int],
        "-cla": ["CustomLossActivate", 0, int],
        "-ni": ["noInit", 4, int],
        "-ei": ["euclideanInit", 0, int],
        "-opc": ["originPrimeCopy", 0, int],
        "-npi": ["noProjectionIntervals", 2, int],
        # Execution Options
        "-td": ["toDo", "all", str],
        "-ts": ["toSkip", "", str],
        "-ds": ["doSave", 0, int],
        "-fp": ["filePath", "", str],
        "-sim": ["computeSIM", 0, int],
    }
    return corr


def getRevParamsCorr():
    corr = getParamsCorr()
    revCorr = {}
    for p in corr:
        revCorr[corr[p][0]] = [p, corr[p][1], corr[p][2]]
    return revCorr


def parseArgs():
    corr = getParamsCorr()
    parser = argparse.ArgumentParser()
    for p in corr:
        parser.add_argument(p, dest=corr[p][0], default=corr[p][1], type=corr[p][2])
    # Get params
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


def getParamStringCorr():
    corr = {
        "PT": ["pt", None],
        "C": ["coef", None],
        "E1": ["eps1", None],
        "E2": ["eps2", None],
        "E3": ["eps3", None],
        "LR": ["learningRate", None],
        "RW": ["reconstructionLossWeight", 1.0],
        "MW": ["metricLossWeight", 0.0],
        "CW": ["clusteringLossWeight", 0.0],
        "CWT": ["clusteringLossTemperature", 5.0],
        "DW": ["customLossDynamicWeight", 0],
        "BS": ["batchSize", None],
        "AF": ["activationFunction", None],
        "OP": ["optimizer", None],
        "MI": ["minIteration", None],
        "MAI": ["maxIteration", 0],
        "NPI": ["noProjectionIntervals", None],
        "NI": ["noInit", None],
        "EI": ["euclideanInit", 0],
        "OPC": ["originPrimeCopy", 0],
        "NL": ["noLayers", None],
        "ING": ["inputNoGeodesics", None],
        "IBSL": ["inputBarySizeLimit", None],
        "LNG": ["latentNoGeodesics", None],
        "LBSL": ["latentBarySizeLimit", None],
        "BSL": ["barySizeLimit", None],
        "A": ["activate", 1],
        "FS": ["fullSymmetricAE", 0],
        "AOI": ["activateOutputInit", 0],
        "NT": ["noThreads", multiprocessing.cpu_count()],
    }
    return corr


def getParamString(args, isLap=None):
    paramString = ""
    if not args["pga"]:
        paramString += "L_" if isLap is None and isLaptop() or isLap else "D_"
        paramString += "PD" if args["isPD"] == 1 else "MT"
        paramStringCorr = getParamStringCorr()
        for param in paramStringCorr:
            val = args[paramStringCorr[param][0]]
            if (
                param != "NT"
                and (
                    paramStringCorr[param][1] is None
                    or val != paramStringCorr[param][1]
                )
            ) or (param == "NT" and val == 1):
                paramString += "_" + param + "_" + str(val)
    else:
        paramString = getMinimalParamString(args)
    return paramString


def getMinimalParamString(args):
    paramString = ""
    paramString += "PD" if args["isPD"] == 1 else "MT"
    paramString += "_PT_" + str(args["pt"])
    paramString += "_C_" + str(args["coef"])
    if not args["isPD"]:
        paramString += "_E1_" + str(args["eps1"])
        paramString += "_E2_" + str(args["eps2"])
        paramString += "_E3_" + str(args["eps3"])
    return paramString


def contains(names, name):
    splitted = re.split(",", names)  # split by / and | and ,
    for split in splitted:
        if split != "" and split in name:
            return True
    return False


def doSkip(args, path):
    if contains(args["toSkip"], path):
        return True
    if not args["toDo"] == "all" and not contains(args["toDo"], path):
        return True
    return False


def putDatasetParams(dirArgs, sys_argv, i):
    corr = getParamsCorr()
    customParams = {corr[p][0]: False for p in corr}
    for arg in sys_argv:
        if arg in corr:
            customParams[corr[arg][0]] = True

    lr, actFun, eucInit, minIteration = getDatasetParams(
        dirArgs["isPD"], customParams["coef"]
    )

    if not customParams["pt"] or dirArgs["pt"] < 0:
        dirArgs["pt"] = pts[i]
        if dirArgs["ptMult"] >= 0:
            dirArgs["pt"] *= dirArgs["ptMult"]
    if not customParams["coef"]:
        dirArgs["coef"] = coefs[i]
    # dirArgs["coef"] = 0
    if not customParams["eps1"]:
        dirArgs["eps1"] = eps1s[i]
    if not customParams["minIteration"]:
        dirArgs["minIteration"] = minIteration[i]
    if not customParams["learningRate"]:
        dirArgs["learningRate"] = lr[i]
    if not customParams["activationFunction"]:
        dirArgs["activationFunction"] = actFun[i]
    if not customParams["euclideanInit"]:
        dirArgs["euclideanInit"] = eucInit[i]


def parseResArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lap", dest="isLaptop", default=isLaptop(), type=int)
    parser.add_argument("-pp", dest="prePath", default="", type=str)
    args, _ = parser.parse_known_args()
    return args
