# MT-AE

## Installation Note

### Install the dependencies

```bash
sudo apt-get install cmake-qt-gui libboost-system-dev libpython3.8-dev libxt-dev libxcursor-dev libopengl-dev
sudo apt-get install qttools5-dev libqt5x11extras5-dev libqt5svg5-dev qtxmlpatterns5-dev-tools 
sudo apt-get install python3-sklearn 
sudo apt-get install libsqlite3-dev 
sudo apt-get install gawk
```

### Install Paraview

First, go in the parent directory of where this repository (`MT-PGA`) is and run the following commands:
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


### Install Torch


### Install TTK

Go in the `ttk-dev2` directory then run the following commands:
(replace the `4` in `make -j4` by the number of available cores on your system)

```bash
mkdir build && cd build
paraviewPath=`pwd`/../../../ttk-paraview/install/lib/cmake/paraview-5.10
cmake -DCMAKE_INSTALL_PREFIX=../install -DParaView_DIR=$paraviewPath ..
make -j4
make -j4 install
```

Stay in the build directory and set the environment variables:
(replace `3.8` in `python3.8` by your version of python)

```bash
TTK_PREFIX=`pwd`/../install
export PV_PLUGIN_PATH=$TTK_PREFIX/bin/plugins/TopologyToolKit
export LD_LIBRARY_PATH=$TTK_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$TTK_PREFIX/lib/python3.10/site-packages
```

### Get the results

Go in the root directory of this repository and extract the data:

```bash
tar xvJf data.tar.xz
```

#### Table 1
