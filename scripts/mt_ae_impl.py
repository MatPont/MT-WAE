from paraview.simple import *
import os
import tempfile
import sys

from argsParser import getParamString, getRevParamsCorr
from dataLoader import load_data


def getCommandParameters(dirArgs):
    command = ""
    for arg in dirArgs:
        val = str(dirArgs[arg])
        if len(val) == 0:
            val = '""'
        command += str(arg) + " " + val + " "
    return command


def saveOutput(tTKMergeTreeAutoencoder1, filePath, isAE=True):
    filePath += "/"
    if not os.path.isdir(filePath):
        os.mkdir(filePath)

    if isAE:
        XMLMultiBlockDataWriter(
            Input=OutputPort(tTKMergeTreeAutoencoder1, 0),
            FileName=filePath + "data.vtm",
        ).UpdatePipeline()
        XMLMultiBlockDataWriter(
            Input=OutputPort(tTKMergeTreeAutoencoder1, 1),
            FileName=filePath + "origins.vtm",
        ).UpdatePipeline()
        XMLMultiBlockDataWriter(
            Input=OutputPort(tTKMergeTreeAutoencoder1, 2),
            FileName=filePath + "coefs.vtm",
        ).UpdatePipeline()
        XMLMultiBlockDataWriter(
            Input=OutputPort(tTKMergeTreeAutoencoder1, 3),
            FileName=filePath + "vectors.vtm",
        ).UpdatePipeline()
    else:
        XMLMultiBlockDataWriter(
            Input=OutputPort(tTKMergeTreeAutoencoder1, 0),
            FileName=filePath + "bary.vtm",
        ).UpdatePipeline()
        XMLTableWriter(
            Input=OutputPort(tTKMergeTreeAutoencoder1, 1),
            FileName=filePath + "coefs.vtt",
        ).UpdatePipeline()
        XMLTableWriter(
            Input=OutputPort(tTKMergeTreeAutoencoder1, 2),
            FileName=filePath + "vectors.vtt",
        ).UpdatePipeline()
        XMLTableWriter(
            Input=OutputPort(tTKMergeTreeAutoencoder1, 2),
            FileName=filePath + "corr.vtt",
        ).UpdatePipeline()


def mt_ae(args):
    input1, input2, info = load_data(args["dirPath"], args["coef"], args["isPD"])
    if input1 is None and input2 is None:
        return

    if args["pga"] == 0:
        # create a new 'TTK MergeTreeAutoencoder'
        tTKMergeTreeAutoencoder1 = TTKMergeTreeAutoencoder(
            registrationName="TTKMergeTreeAutoencoder1",
            Input=input1,
            OptionalInput=input2,
            Info=info,
        )
        # Input Options
        tTKMergeTreeAutoencoder1.DebugLevel = args["debugLevel"]
        tTKMergeTreeAutoencoder1.UseAllCores = 0
        tTKMergeTreeAutoencoder1.ThreadNumber = args["noThreads"]
        tTKMergeTreeAutoencoder1.Epsilon1 = args["eps1"]
        tTKMergeTreeAutoencoder1.Epsilon2 = args["eps2"]
        tTKMergeTreeAutoencoder1.Epsilon3 = args["eps3"]
        tTKMergeTreeAutoencoder1.NormalizedWasserstein = not args["isPD"]
        tTKMergeTreeAutoencoder1.PairTypeMixtureCoefficient = args["coef"]
        tTKMergeTreeAutoencoder1.PersistenceThreshold = args["pt"]
        tTKMergeTreeAutoencoder1.Deterministic = args["deterministic"]
        # Architecture Options
        tTKMergeTreeAutoencoder1.NumberofEncoderLayers = args["noLayers"]
        tTKMergeTreeAutoencoder1.BarycenterSizeLimitPercent = args["barySizeLimit"]
        tTKMergeTreeAutoencoder1.InputNumberofGeodesics = args["inputNoGeodesics"]
        tTKMergeTreeAutoencoder1.InputOriginPrimeSizePercent = args[
            "inputBarySizeLimit"
        ]
        tTKMergeTreeAutoencoder1.LatentSpaceNumberofGeodesics = args[
            "latentNoGeodesics"
        ]
        tTKMergeTreeAutoencoder1.LatentSpaceOriginPrimeSizePercent = args[
            "latentBarySizeLimit"
        ]
        tTKMergeTreeAutoencoder1.Activate = args["activate"]
        tTKMergeTreeAutoencoder1.ActivationFunction = args["activationFunction"]
        tTKMergeTreeAutoencoder1.FullSymmetricAE = args["fullSymmetricAE"]
        tTKMergeTreeAutoencoder1.ActivateOutputInit = args["activateOutputInit"]
        # Optimization Options
        tTKMergeTreeAutoencoder1.MinIteration = args["minIteration"]
        tTKMergeTreeAutoencoder1.MaxIteration = args["maxIteration"]
        tTKMergeTreeAutoencoder1.IterationGap = args["iterationGap"]
        tTKMergeTreeAutoencoder1.BatchSize = args["batchSize"]
        tTKMergeTreeAutoencoder1.Optimizer = args["optimizer"]
        tTKMergeTreeAutoencoder1.GradientStepSize = args["learningRate"]
        tTKMergeTreeAutoencoder1.Beta1 = args["beta1"]
        tTKMergeTreeAutoencoder1.Beta2 = args["beta2"]
        tTKMergeTreeAutoencoder1.ReconstructionLossWeight = args[
            "reconstructionLossWeight"
        ]
        tTKMergeTreeAutoencoder1.MetricLossWeight = args["metricLossWeight"]
        tTKMergeTreeAutoencoder1.ClusteringLossWeight = args["clusteringLossWeight"]
        tTKMergeTreeAutoencoder1.ClusteringLossTemperature = args[
            "clusteringLossTemperature"
        ]
        tTKMergeTreeAutoencoder1.CustomLossDynamicWeight = args[
            "customLossDynamicWeight"
        ]
        tTKMergeTreeAutoencoder1.CustomLossSpace = args["customLossSpace"]
        tTKMergeTreeAutoencoder1.CustomLossActivate = args["CustomLossActivate"]
        tTKMergeTreeAutoencoder1.Numberofinit = args["noInit"]
        tTKMergeTreeAutoencoder1.EuclideanVectorsInit = args["euclideanInit"]
        tTKMergeTreeAutoencoder1.InitOriginPrimeStructByCopy = args["originPrimeCopy"]
        tTKMergeTreeAutoencoder1.NumberOfProjectionIntervals = args[
            "noProjectionIntervals"
        ]
        # tTKMergeTreeAutoencoder1.NodePerTask = 4

        tTKMergeTreeAutoencoder1.CreateOutput = (
            args["doSave"] == 1
            or args["metricLossWeight"] != 0
            or args["computeSIM"] == 1
        )

        # ---
        tTKMergeTreeAutoencoder1.UpdatePipeline()

        if args["doSave"]:
            saveOutput(tTKMergeTreeAutoencoder1, args["filePath"])
    else:
        # create a new 'TTK MergeTreePrincipalGeodesics'
        tTKMergeTreePrincipalGeodesics = TTKMergeTreePrincipalGeodesics(
            Input=input1, OptionalInput=input2
        )
        # Input Options
        tTKMergeTreePrincipalGeodesics.DebugLevel = args["debugLevel"]
        tTKMergeTreePrincipalGeodesics.UseAllCores = 0
        tTKMergeTreePrincipalGeodesics.ThreadNumber = args["noThreads"]
        tTKMergeTreePrincipalGeodesics.Epsilon1 = args["eps1"]
        tTKMergeTreePrincipalGeodesics.Epsilon2 = args["eps2"]
        tTKMergeTreePrincipalGeodesics.Epsilon3 = args["eps3"]
        tTKMergeTreePrincipalGeodesics.NormalizedWasserstein = not args["isPD"]
        tTKMergeTreePrincipalGeodesics.PairTypeMixtureCoefficient = args["coef"]
        tTKMergeTreePrincipalGeodesics.PersistenceThreshold = args["pt"]
        tTKMergeTreePrincipalGeodesics.Deterministic = args["deterministic"]
        # sizeLimit = args["barySizeLimit"] * args["latentBarySizeLimit"] / 100.0
        sizeLimit = args["latentBarySizeLimit"]
        print("sizeLimit =", sizeLimit)
        tTKMergeTreePrincipalGeodesics.BarycenterSizeLimitPercent = sizeLimit
        tTKMergeTreePrincipalGeodesics.NumberOfGeodesics = args["latentNoGeodesics"]
        """tTKMergeTreePrincipalGeodesics.NumberOfProjectionIntervals = args[
            "noProjectionIntervals"
        ]"""
        tTKMergeTreePrincipalGeodesics.NumberOfProjectionIntervals = 16
        tTKMergeTreePrincipalGeodesics.NumberOfProjectionSteps = 8

        # ---
        tTKMergeTreePrincipalGeodesics.UpdatePipeline()

        if args["doSave"]:
            saveOutput(tTKMergeTreePrincipalGeodesics, args["filePath"], False)


def execute(args):
    #####################
    # Print parameters
    #####################
    print("#" * 80, flush=True)
    print(getParamString(args))
    command = "mt_ae_impl.py "
    command += getCommandParameters(args)
    print("#" * 80, flush=True)
    print(command, flush=True)
    print("#" * 80, flush=True)
    maxLen = 0
    for arg in args:
        maxLen = max(maxLen, len(arg))
    print("#" * 80, flush=True)
    for arg in args:
        if arg == "dirPath":
            print("####### Input Options", flush=True)
        elif arg == "noLayers":
            print("####### Architecture Options", flush=True)
        elif arg == "minIteration":
            print("####### Optimization Options", flush=True)
        elif arg == "toDo":
            print("####### Execution Options", flush=True)
        print(
            arg
            + " " * (1 - int(len(arg) / maxLen))
            + "." * (maxLen - len(arg) - 1)
            + " = "
            + str(args[arg]),
            flush=True,
        )
    print("#" * 80, flush=True)

    #####################
    # Run parameters
    #####################
    mt_ae(args)


if __name__ == "__main__":
    #####################
    # Load parameters
    #####################
    revCorr = getRevParamsCorr()
    args = {}
    for i in range(1, len(sys.argv), 2):
        fun = revCorr[sys.argv[i]][2]
        args[sys.argv[i]] = fun(sys.argv[i + 1])

    #####################
    # Run parameters
    #####################
    execute(args)
