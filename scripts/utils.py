prepath = "../data/"

paths = [
    "startingVortexGoodEnsemble_TA/",  # 0
    "isabella_velocity_goodEnsemble_TA/",  # 1
    "seaSurfaceHeightGoodEnsemble_TA/",  # 2
    "vortexStreetGoodEnsemble2_TA/",  # 3
    "particularEnsemble_TA/",  # 4
    "cloud5_TA/",  # 5
    "astroTurbulence_TA/",  # 6
    "impactEnsemble3CTev_TA/",  # 7
    "darkSky100S_TA/",  # 8
    "volcanic2_TA/",  # 9
    "astro3DTurbulence_TA/",  # 10
    "earthquake2_TA/",  # 11
    "darkSky4_100SumST_TA/",  # 12
]

order = [9, 8, 10, 11, 2, 1, 5, 0, 3, 4, 6, 7, 3]

names = [
    "Asteroid Impact (3D)",
    "Cloud processes (2D)",
    "Viscous fingering (3D)",
    "Dark matter (3D)",
    "Volcanic eruptions (2D)",
    "Ionization front (2D)",
    "Ionization front (3D)",
    "Earthquake (3D)",
    "Isabel (3D)",
    "Starting Vortex (2D)",
    "Sea Surface Height (2D)",
    "Vortex Street (2D)",
]

noData = [7, 12, 15, 40, 12, 16, 16, 12, 12, 12, 48, 45]

# cardinal = [1295, 1209, 118, 2592, 811, 135, 763, 1203, 1338, 124, 1787, 23]
cardinal = [1295, 1209, 118, 316, 811, 135, 763, 1203, 1338, 124, 1787, 23]

for i in range(len(paths)):
    paths[i] = prepath + paths[i]

defaultPt = 0.25
pts = [
    defaultPt,
    1.0,
    1.0,
    defaultPt,
    defaultPt,
    10.0,
    defaultPt,
    defaultPt,
    10.0,
    defaultPt,
    1.0,
    defaultPt,
    10.0,
]
coefs = [0.5, 0.5, 0.75, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
defaultEps1 = 5.0
eps1s = [
    defaultEps1,
    defaultEps1,
    defaultEps1,
    defaultEps1,
    defaultEps1,
    10.0,
    defaultEps1,
    defaultEps1,
    defaultEps1,
    defaultEps1,
    defaultEps1,
    defaultEps1,
    defaultEps1,
]


def getDatasetName(dataset):
    return dataset.split("_TA")[-2].split("/")[-1]


def isLaptop():
    return False
