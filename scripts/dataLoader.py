from paraview.simple import *
import os


def cdbRead(fileName):
    # create a new 'TTK CinemaReader'
    tTKCinemaReader1 = TTKCinemaReader(DatabasePath=fileName)

    # create a new 'TTK CinemaProductReader'
    tTKCinemaProductReader1 = TTKCinemaProductReader(Input=tTKCinemaReader1)
    tTKCinemaProductReader1.AddFieldDataRecursively = 1

    # create a new 'TTK FlattenMultiBlock'
    tTKFlattenMultiBlock1 = TTKFlattenMultiBlock(Input=tTKCinemaProductReader1)

    # create a new 'Programmable Filter'
    programmableFilter1 = ProgrammableFilter(Input=tTKFlattenMultiBlock1)
    programmableFilter1.Script = """outBlocks = vtk.vtkMultiBlockDataSet()
inp = self.GetInputDataObject(0, 0)
for i in range(inp.GetBlock(0).GetNumberOfBlocks()):
    outBlocks.SetBlock(i, vtk.vtkMultiBlockDataSet())
for b in range(inp.GetNumberOfBlocks()):
  for i in range(inp.GetBlock(b).GetNumberOfBlocks()):
    outBlocks.GetBlock(i).SetBlock(b, inp.GetBlock(b).GetBlock(i))
output = self.GetOutputDataObject(0)
output.ShallowCopy(outBlocks)"""
    programmableFilter1.RequestInformationScript = ""
    programmableFilter1.RequestUpdateExtentScript = ""
    programmableFilter1.PythonPath = ""

    return programmableFilter1


def labelLoader(fileName):
    # create a new 'CSV Reader'
    labelcsv = CSVReader(FileName=[fileName])
    labelcsv.HaveHeaders = 0

    # create a new 'Calculator'
    calculator1 = Calculator(Input=labelcsv)
    calculator1.AttributeType = "Row Data"
    calculator1.ResultArrayName = "ClusterAssignment"
    calculator1.Function = '"Field 0"'

    return calculator1


def load_data(dirPath, coef, isPD):
    input1, input2 = None, None

    files = os.listdir(dirPath)
    stFilePath = [a for a in files if "ST_light.cdb" in a]
    if len(stFilePath) != 0:
        stFilePath = dirPath + stFilePath[0]
        dataST = cdbRead(stFilePath)
    else:
        print("[ERROR] can not find jtFilePath or stFilePath")
        return None, None, None

    input1 = dataST

    label = None
    labelFile = [a for a in files if "label.csv" in a]
    if len(labelFile) != 0:
        label = labelLoader(dirPath + labelFile[0])

    return input1, input2, label


def load_dataFromArgs(args):
    return load_data(args["dirPath"], args["coef"], args["isPD"])
