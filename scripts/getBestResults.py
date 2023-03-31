import sys
from utils import *
import argsParser

import os


def getOutputDir(dataset, isPGA=None, prePath=None):
    dirName = "./outFiles/"
    if isPGA is None:
        isPGA = vars(argsParser.parseArgs())["pga"]
    if isPGA:
        dirName += "PGA/"
    if prePath is None:
        prePath = vars(argsParser.parseResArgs())["prePath"]
    dirName += prePath + "/"
    if not os.path.isdir(dirName):
        os.makedirs(dirName)
    datasetName = getDatasetName(dataset)
    dirName += datasetName + "/"
    return dirName


def getMetricsFromOut(out, isPGA=None):
    if isPGA is None:
        isPGA = vars(argsParser.parseArgs())["pga"]
    if not isPGA:
        loss = float(out.split("Best loss is")[-1].split("(")[0])
        if "RUN" in out:
            loss = float(out.split("- Rec. loss")[-1].split("=")[1].split("\n")[0])
        time = float(out.split("at time")[-1].split("\n")[0])
    else:
        loss = float(out.split("Best energy is")[-1].split("(")[0])
        time = float(out.split("Total time")[1].split("s")[0].split("[")[-1])
    iteration = int(out.split("iteration")[-1].split("/")[0])
    return loss, time, iteration


def getMetricsFromFile(fileName, isPGA=None):
    f = open(fileName, "r")
    loss, time, iteration = getMetricsFromOut(f.read(), isPGA=isPGA)
    f.close()
    return loss, time, iteration


def filterFiles(
    files,
    outputDir,
    dirArgs,
    resArgs,
    filterLap=True,
    filterThread=True,
    filterFilesT=True,
    filterDir=False,
    filterCoef=False,
    filterEps=False,
    filterOut=False,
):
    files = [
        f
        for f in files
        if ("PD" in f and dirArgs["isPD"]) or ("MT" in f and not dirArgs["isPD"])
    ]
    if filterLap:
        files = [
            f
            for f in files
            if (f[0] == "L" and resArgs["isLaptop"])
            or (f[0] == "D" and not resArgs["isLaptop"])
        ]
    if filterThread:
        files = [
            f
            for f in files
            if (not "NT_1" in f and dirArgs["noThreads"] != 1)
            or ("NT_1" in f and dirArgs["noThreads"] == 1)
        ]
    if "onlyRec" in dirArgs and dirArgs["onlyRec"] == 1:
        files = [f for f in files if "MW" not in f and "CW" not in f]
    if "onlyMetric" in dirArgs and dirArgs["onlyMetric"] == 1:
        # files = [f for f in files if "MW" in f]
        files = [f for f in files if "MW" in f and not "CW" in f]
    if "onlyClust" in dirArgs and dirArgs["onlyClust"] == 1:
        # files = [f for f in files if "CW" in f]
        files = [f for f in files if "CW" in f and not "MW" in f]
    if "onlyMetricClust" in dirArgs and dirArgs["onlyMetricClust"] == 1:
        files = [f for f in files if "MW" in f and "CW" in f]
    if filterFilesT and not filterDir:
        files = [f for f in files if not os.path.isdir(outputDir + f)]
    if filterDir and not filterFilesT:
        files = [f for f in files if os.path.isdir(outputDir + f)]
    if filterCoef:
        files = [f for f in files if "_C_" + str(dirArgs["coef"]) in f]
    if filterEps and not dirArgs["isPD"]:
        files = [
            f
            for f in files
            if "_E1_" + str(dirArgs["eps1"]) in f
            and "_E2_" + str(dirArgs["eps2"]) in f
            and "_E3_" + str(dirArgs["eps3"]) in f
        ]
    if filterOut:
        files = [f for f in files if ".out" in f]
    return files


def getResultsFromPath(outputDir, dirArgs, resArgs):
    args = dirArgs.copy()

    files = os.listdir(outputDir)
    # filterLap = not args["pga"]
    filterLap = "-lap" in sys.argv
    # filterOut = args["pga"]
    filterOut = True

    if args["metricLossWeight"] != 0 and args["clusteringLossWeight"] != 0:
        args["onlyMetricClust"] = 1
    elif args["metricLossWeight"] != 0 and args["clusteringLossWeight"] == 0:
        args["onlyMetric"] = 1
    elif args["metricLossWeight"] == 0 and args["clusteringLossWeight"] != 0:
        args["onlyClust"] = 1
    elif args["reconstructionLossWeight"] != 0:
        args["onlyRec"] = 1

    files = filterFiles(
        files,
        outputDir,
        args,
        resArgs,
        filterLap=filterLap,
        filterOut=filterOut,
        filterCoef=True,
        filterEps=True,
        filterThread=False,
    )

    allLosses = []
    for f in files:
        loss, time, iteration = getMetricsFromFile(outputDir + f)
        allLosses.append([loss, f, time, iteration])
    allLosses = sorted(allLosses)
    return allLosses


def getBestResultFromPath(outputDir, dirArgs, resArgs):
    res = getResultsFromPath(outputDir, dirArgs, resArgs)
    return res[0] if len(res) > 0 else [None, None, None]


def getParamFromFileName(fileName, param):
    if param not in fileName:
        corr = argsParser.getParamStringCorr()
        return corr[param][1]
    return fileName.split(param)[1].split("_")[1]


if __name__ == "__main__":
    resArgs = vars(argsParser.parseResArgs())
    dirArgs = vars(argsParser.parseArgs())
    for i in range(len(paths)):
        dataset = paths[i]
        print("#" * 80, flush=True)
        print(dataset, flush=True)
        print("#" * 80, flush=True)

        outputDir = getOutputDir(dataset)
        if not os.path.isdir(outputDir):
            continue
        argsParser.putDatasetParams(dirArgs, sys.argv, i)
        allLosses = getResultsFromPath(outputDir, dirArgs, resArgs)

        if len(allLosses) == 0:
            continue

        print("Best loss =", allLosses[0][0])
        print("- at time =", allLosses[0][2])
        print("- at iter =", allLosses[0][3])

        bestParams = {param: None for param in ["LR", "AF", "EI"]}
        for param in bestParams:
            bestParams[param] = getParamFromFileName(allLosses[0][1], param)
        print("Best params:")
        for param in bestParams:
            print(" -", param, ":", bestParams[param])
        f = open(outputDir + allLosses[0][1], "r")
        content = f.read()
        f.close()
        minIteration = content.split("minIteration")[1].split(" ")[1]
        print(minIteration)

        for res in allLosses:
            print(res)
