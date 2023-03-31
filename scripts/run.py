import sys
import os

from utils import *
from hyperParameters import *
import argsParser
from mt_ae_impl import getCommandParameters, execute
from getBestResults import getOutputDir, getMetricsFromFile, getMetricsFromOut


def getNewRunID(dataset, paramString):
    dirName = getOutputDir(dataset)
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    rID = 0
    runID = "_RUN_" + str(rID)
    fileName = assembleFileName(dirName, paramString, runID)
    while os.path.isfile(fileName):
        rID = rID + 1
        runID = "_RUN_" + str(rID)
        fileName = assembleFileName(dirName, paramString, runID)
    return runID


def assembleFileName(dirName, paramString, runID=""):
    return dirName + paramString + runID + ".out"


def getOutputFileName(dataset, paramString, runID="", isPGA=None, prePath=None):
    dirName = getOutputDir(dataset, isPGA=isPGA, prePath=prePath)
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    fileName = assembleFileName(dirName, paramString, runID)
    return fileName


if __name__ == "__main__":
    args = argsParser.parseArgs()
    args = vars(args)

    for i in range(len(paths)):
        if argsParser.doSkip(args, paths[i]):
            continue

        args["dirPath"] = paths[i]
        argsParser.putDatasetParams(args, sys.argv, i)

        # Prepare fileName
        paramString = argsParser.getParamString(args)
        print(paramString)
        nonDeterministic = (
            not args["deterministic"]
            or args["metricLossWeight"] != 0
            or args["clusteringLossWeight"] != 0
        )
        runID = ""
        if nonDeterministic:
            runID = getNewRunID(paths[i], paramString)
        fileName = getOutputFileName(paths[i], paramString, runID=runID)
        if nonDeterministic or args["doSave"] == 1:
            args["doSave"] = 1
            args["filePath"] = fileName.replace(".out", "")

        # Execute
        tempFileName = args["toDo"] + "_PD_" + str(args["isPD"])
        if os.path.isfile(fileName):
            print("read ", fileName)
            loss, time, _ = getMetricsFromFile(fileName)
        else:
            execute(args)
            if args["noThreads"] == 1:
                tempFileName += "_NT_" + str(args["noThreads"])
            loss, time, _ = getMetricsFromFile(tempFileName)
        print("loss =", loss, flush=True)
        print("time =", time, flush=True)

        if not os.path.isfile(fileName) and os.path.isfile(tempFileName):
            os.rename(tempFileName, fileName)
