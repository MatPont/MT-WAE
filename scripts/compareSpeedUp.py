try:
    from prettytable import PrettyTable

    prettyTableImported = True
except:
    prettyTableImported = False
import sys
import os

from utils import *
import argsParser
from run import getOutputFileName
from getBestResults import getMetricsFromFile


def putDataRedParams(args, pga, isPD, nt):
    lrMT = [
        0.05,
        0.0025,
        0.0025,
        0.1,
        0.025,
        0.1,
        0.001,
        0.025,
        0.005,
        0.01,
        0.01,
        0.025,
    ]
    lrPD = [
        0.5,
        0.1,
        0.001,
        0.05,
        10.0,
        100.0,
        0.005,
        0.00025,
        0.0025,
        0.005,
        0.01,
        0.5,
    ]
    namesT = [
        "starting",
        "isabel",
        "sea",
        "vortex",
        "particular",
        "cloud",
        "astroT",
        "impact",
        "volcanic",
        "astro3D",
        "earthquake",
        "darkSky4",
    ]
    lbsl = [10.0, 10.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    miMT = [1000, 500, 100, 500, 1500, 500, 500, 500, 500, 100, 500, 500]
    miPD = [1000, 200, 100, 500, 1000, 500, 500, 500, 500, 100, 500, 500]
    mai = [0, 0, 500, 0, 0, 0, 0, 0, 0, 500, 0, 0]
    ni = [4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    eiMT = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    eiPD = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    opc = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    bsl = 10.0
    ing = 3
    lng = 3
    af = 1

    args["prePath"] = "dataRed"
    args["pga"] = pga
    args["isPD"] = isPD
    args["noThreads"] = nt
    index = [i for i in range(len(namesT)) if namesT[i] in args["dirPath"]]
    index = index[0]
    args["learningRate"] = lrPD[index] if args["isPD"] else lrMT[index]
    args["latentBarySizeLimit"] = lbsl[index]
    args["minIteration"] = miPD[index] if args["isPD"] else miMT[index]
    args["maxIteration"] = mai[index]
    args["noInit"] = ni[index]
    args["euclideanInit"] = eiPD[index] if args["isPD"] else eiMT[index]
    args["originPrimeCopy"] = opc[index]
    args["barySizeLimit"] = bsl
    args["inputNoGeodesics"] = ing
    args["latentNoGeodesics"] = lng
    args["activationFunction"] = af


def getResults(dirArgs, resArgs):
    args = dirArgs.copy()

    res = []
    for nt in [dirArgs["noThreads"], 1]:
        for pga in [0]:
            for isPD in [0, 1]:
                putDataRedParams(args, pga, isPD, nt)

                paramString = argsParser.getParamString(args, isLap=resArgs["isLaptop"])
                fileName = getOutputFileName(
                    args["dirPath"],
                    paramString,
                    isPGA=args["pga"],
                    prePath=args["prePath"],
                )
                if not os.path.isfile(fileName):
                    print("CAN NOT read", fileName)
                    res.append(None)
                    continue
                print("read", fileName)
                loss, time, iteration = getMetricsFromFile(fileName, isPGA=args["pga"])
                res.append(time)
    return res


if __name__ == "__main__":
    dirArgs = vars(argsParser.parseArgs())
    resArgs = vars(argsParser.parseResArgs())

    tableValues = []
    for i in range(len(paths)):
        if argsParser.doSkip(dirArgs, paths[i]):
            continue
        if "darkSky100S" in paths[i]:
            continue

        print("#" * 80, flush=True)
        print(paths[i], flush=True)
        print("#" * 80, flush=True)
        dirArgs["dirPath"] = paths[i]
        argsParser.putDatasetParams(dirArgs, sys.argv, i)

        res = getResults(dirArgs, resArgs)
        # MT-WAE ; PD-WAE ; MT-WAE-SEQ ; PD-WAE-SEQ

        speedupMT = (
            res[2] / res[0] if res[2] is not None and res[0] is not None else None
        )
        speedupPD = (
            res[3] / res[1] if res[3] is not None and res[1] is not None else None
        )
        values = [res[3], res[1], speedupPD, res[2], res[0], speedupMT]
        for j in range(len(values)):
            if values[j] is not None:
                values[j] = round(values[j], 2)
        datasetName = getDatasetName(paths[i])
        tableValues.append([datasetName] + values)

    # Table
    if prettyTableImported:
        table = PrettyTable()
        table.field_names = [
            "Dataset",
            "PD time 1c",
            "PD time 20c",
            "PT Speedup",
            "MT time 1c",
            "MT time 20c",
            "MT Speedup",
        ]
        table.align["Dataset"] = "l"
        for i in range(1, len(table.field_names)):
            table.align[table.field_names[i]] = "r"
        table.add_rows(tableValues)
        print()
        print(table)

    # Latex table
    reordered = [[] for i in range(len(tableValues))]
    for i in range(len(tableValues)):
        index = [j for j in range(len(paths)) if tableValues[i][0][:-1] in paths[j]][0]
        reordered[order[index]] = [
            names[order[index]],
            noData[order[index]],
            f"{cardinal[order[index]]:,}",
        ] + [
            f"{x:,.2f}" if not isinstance(x, str) and x is not None else x
            for x in tableValues[i][1:]
        ]

    print("\\begin{table}")
    print(
        "\\caption{Running times (in seconds) of our algorithm for PD-WAE and MT-WAE computation (first sequential, then with 20 cores).}"
    )
    print(
        "% architecture like data reduction, 2 layers, N1 = 10 and dmax = 3 for each layer"
    )
    print("\\label{tab_timings}")
    print("\\centering")
    print("\\scalebox{0.8}{")
    print("  \\begin{tabular}{|l|r|r||r|r|r||r|r|r|}")
    print("    \\hline")
    print(
        "    \\rule{0pt}{2.25ex} \\textbf{Dataset} & $\\ensembleSize$ & $|\\branchtree|$ & \\multicolumn{3}{c||}{PD-WAE} & \\multicolumn{3}{c|}{MT-WAE} \\\\"
    )
    print("    & & & 1 c. & 20 c. & Speedup & 1 c. & 20 c. & Speedup \\\\")
    print("    \\hline")
    # %% numbers here
    for val in reordered:
        print("      " + " & ".join([str(x) for x in val]) + " \\\\")
    print("    \\hline")
    print("  \\end{tabular}")
    print("}")
    print("\\end{table}")
