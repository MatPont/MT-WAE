# MT-WAE

This github repository contains the exact code used for the reference below.

## Reference

If you plan to use this code to generate results for a scientific document, thanks for referencing the following publication:

"Wasserstein Auto-Encoders of Merge Trees (and Persistence Diagrams)"  
Mathieu Pont, Julien Tierny  
IEEE Transactions on Visualization and Computer Graphics, 2023.  

[Paper](https://arxiv.org/pdf/2307.02509.pdf)

## Installation Note

Tested on Ubuntu 22.04.3 LTS.

### Install the dependencies

```bash
sudo apt-get install cmake-qt-gui libboost-system-dev libpython3.10-dev libxt-dev libxcursor-dev libopengl-dev
sudo apt-get install qttools5-dev libqt5x11extras5-dev libqt5svg5-dev qtxmlpatterns5-dev-tools 
sudo apt-get install python3-sklearn 
sudo apt-get install libsqlite3-dev 
sudo apt-get install gawk
sudo apt-get install git
```

### Install Paraview

First, go in the root of this repository and run the following commands:
(replace the `4` in `make -j4` by the number of available cores on your system)

```bash
git clone https://github.com/topology-tool-kit/ttk-paraview.git
cd ttk-paraview
git checkout 5.10.1
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPARAVIEW_USE_PYTHON=ON -DPARAVIEW_INSTALL_DEVELOPMENT_FILES=ON -DCMAKE_INSTALL_PREFIX=../install ..
make -j4
make -j4 install
```

Some warnings are expected when using the `make` command, they should not cause any problems.

Stay in the build directory and set the environment variables:
(replace `3.10` in `python3.10` by your version of python)

```bash
PV_PREFIX=`pwd`/../install
export PATH=$PATH:$PV_PREFIX/bin
export LD_LIBRARY_PATH=$PV_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PV_PREFIX/lib/python3.10/site-packages
```

### Download Torch

Go in the root of this repository and run the following commands:

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip
```

### Install TTK

Go in the `ttk-dev2` directory then run the following commands:
(replace the `4` in `make -j4` by the number of available cores on your system)

```bash
mkdir build && cd build
paraviewPath=`pwd`/../../ttk-paraview/install/lib/cmake/paraview-5.10
torchPath=`pwd`/../../libtorch/share/cmake/Torch/
cmake -DCMAKE_INSTALL_PREFIX=../install -DParaView_DIR=$paraviewPath -DTorch_DIR=$torchPath ..
make -j4
make -j4 install
```

Stay in the build directory and set the environment variables:
(replace `3.10` in `python3.10` by your version of python)

```bash
TTK_PREFIX=`pwd`/../install
export PV_PLUGIN_PATH=$TTK_PREFIX/bin/plugins/TopologyToolKit
export LD_LIBRARY_PATH=$TTK_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$TTK_PREFIX/lib/python3.10/site-packages
```

### Get the results

Go in the root directory of this repository and extract the data:

```bash
tar xvJf data.tar.xz
```

#### Table 1

To reproduce the results of the time table in the paper, please go in the `scripts` directory and enter the following commands:

```bash
for f in *.sh; do chmod u+x $f; done
```

**Run the experiments and print table:**

(replace `N` with the number of available cores on your system)

**To decrease computation time** you can set the optional parameter `ptMult` to a value greater than 1. It will have the effect to multiply the persistence thresholds by `ptMult` and hence decreasing the computation time (for example, replace `[ptMult]` by `7`, default value is `1`). However, the computation time will not decrease the same way for each dataset since the number of pairs removed is not linearly correlated with the persistence threshold. Moreover, when increasing the persistence threshold, the speedup will be lower. Finally, the hyper-parameters in the scripts are optimized for the default persistence thresholds (when `ptMult` equals 1).

**To save output** (not needed to reproduce the time table) you can set the optional parameter `saveOutput` to 1 (default value is 0), it will saves the output of the algorithm (trees/diagrams at each layer including latent space with their coefficients, origins and vectors of each layer) on the disk. You must pass a value for `ptMult` (default value is 1) if you want to use this option.

```bash
./automataSpeedUp.sh N [ptMult] [saveOutput]
```

**To print the results** you can use the following command: 

(if you have used the `ptMult` parameter you should pass it to the script with `-ptMult ptMultVal` where `ptMultVal` is the value you used to run the experiments)

It will print the latex table, but if you want a nice formatting in the console you can install `prettytable` (with `pip install prettytable`) and it will also print the formatted table in the console.

```bash
python3 compareSpeedUp.py -c 0 [-ptMult ptMultVal]
```
