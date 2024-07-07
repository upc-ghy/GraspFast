
# GraspFast
### GraspFast: Multi-stage Lightweight 6-DoF Grasp Pose Fast Detection with RGB-D Image


![GraspFast](https://github.com/upc-ghy/GraspFast/assets/133858812/cc4ba68f-70b2-474b-b29e-30f9fb565988)


## Algorithm Training Requirements

### 1. Algorithm Training Environment
~~~shell
[HaiyuanGui@master01 ~]$ conda list
# packages in environment at /hpcfiles/users/HaiyuanGui/.conda/envs/GraspFast:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main    defaults
_openmp_mutex             5.1                       1_gnu    defaults
absl-py                   1.4.0                    pypi_0    pypi
addict                    2.4.0                    pypi_0    pypi
aiofiles                  22.1.0                   pypi_0    pypi
aiosqlite                 0.19.0                   pypi_0    pypi
anyio                     3.7.1                    pypi_0    pypi
appdirs                   1.4.4                    pypi_0    pypi
argon2-cffi               21.3.0                   pypi_0    pypi
argon2-cffi-bindings      21.2.0                   pypi_0    pypi
arrow                     1.2.3                    pypi_0    pypi
attrs                     23.1.0                   pypi_0    pypi
autolab-core              1.1.1                    pypi_0    pypi
autolab-perception        1.0.0                    pypi_0    pypi
babel                     2.12.1                   pypi_0    pypi
backcall                  0.2.0              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
beautifulsoup4            4.12.2                   pypi_0    pypi
blas                      1.0                         mkl    defaults
bleach                    6.0.0                    pypi_0    pypi
brotlipy                  0.7.0           py37h27cfd23_1003    defaults
bzip2                     1.0.8                h7b6447c_0    defaults
ca-certificates           2023.01.10           h06a4308_0    anaconda
cached-property           1.5.2                    pypi_0    pypi
cachetools                4.2.4                    pypi_0    pypi
certifi                   2022.12.7        py37h06a4308_0    anaconda
cffi                      1.15.1           py37h5eee18b_3    defaults
charset-normalizer        2.0.4              pyhd3eb1b0_0    defaults
click                     8.1.5                    pypi_0    pypi
colorlog                  6.7.0                    pypi_0    pypi
cryptography              39.0.1           py37h9ce1e76_0    defaults
cuda-cudart               11.7.99                       0    nvidia
cuda-cupti                11.7.101                      0    nvidia
cuda-libraries            11.7.1                        0    nvidia
cuda-nvrtc                11.7.99                       0    nvidia
cuda-nvtx                 11.7.91                       0    nvidia
cuda-runtime              11.7.1                        0    nvidia
cvxopt                    1.3.1                    pypi_0    pypi
cycler                    0.11.0                   pypi_0    pypi
cython                    0.29.36                  pypi_0    pypi
debugpy                   1.6.7                    pypi_0    pypi
decorator                 5.1.1              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
defusedxml                0.7.1              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
deprecation               2.1.0                    pypi_0    pypi
dill                      0.3.6                    pypi_0    pypi
docker-pycreds            0.4.0                    pypi_0    pypi
easydict                  1.10                     pypi_0    pypi
entrypoints               0.4              py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
exceptiongroup            1.1.2                    pypi_0    pypi
fastjsonschema            2.17.1                   pypi_0    pypi
ffmpeg                    4.3                  hf484d3e_0    pytorch
ffmpeg-python             0.2.0                    pypi_0    pypi
fonttools                 4.38.0                   pypi_0    pypi
fqdn                      1.5.1                    pypi_0    pypi
freetype                  2.12.1               h4a9f257_0    defaults
future                    0.18.3                   pypi_0    pypi
giflib                    5.2.1                h5eee18b_3    defaults
gitdb                     4.0.10                   pypi_0    pypi
gitpython                 3.1.32                   pypi_0    pypi
gmp                       6.2.1                h295c915_3    defaults
gnutls                    3.6.15               he1e5248_0    defaults
google-auth               1.35.0                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
grasp-nms                 1.0.2                    pypi_0    pypi
graspnetapi               1.2.10                   pypi_0    pypi
grpcio                    1.56.0                   pypi_0    pypi
h5py                      3.8.0                    pypi_0    pypi
idna                      3.4              py37h06a4308_0    defaults
imageio                   2.31.1                   pypi_0    pypi
importlib-metadata        6.7.0                    pypi_0    pypi
importlib-resources       5.12.0                   pypi_0    pypi
importlib_metadata        4.11.3               hd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
importlib_resources       5.2.0              pyhd3eb1b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
intel-openmp              2021.4.0          h06a4308_3561    defaults
ipykernel                 6.16.2                   pypi_0    pypi
ipython                   7.34.0                   pypi_0    pypi
ipython_genutils          0.2.0              pyhd3eb1b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ipywidgets                8.0.7                    pypi_0    pypi
isoduration               20.11.0                  pypi_0    pypi
jedi                      0.18.2                   pypi_0    pypi
jinja2                    3.1.2            py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
joblib                    1.3.1                    pypi_0    pypi
jpeg                      9e                   h5eee18b_1    defaults
json5                     0.9.14                   pypi_0    pypi
jsonpointer               2.4                      pypi_0    pypi
jsonschema                4.17.3           py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
jupyter-core              4.12.0                   pypi_0    pypi
jupyter-events            0.6.3                    pypi_0    pypi
jupyter-packaging         0.12.3                   pypi_0    pypi
jupyter-server            1.24.0                   pypi_0    pypi
jupyter-server-fileid     0.9.0                    pypi_0    pypi
jupyter-server-ydoc       0.8.0                    pypi_0    pypi
jupyter-ydoc              0.2.4                    pypi_0    pypi
jupyter_client            7.4.9            py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
jupyter_core              4.11.2           py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
jupyter_server            1.23.4           py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
jupyterlab                3.6.5                    pypi_0    pypi
jupyterlab-pygments       0.2.2                    pypi_0    pypi
jupyterlab-server         2.23.0                   pypi_0    pypi
jupyterlab-widgets        3.0.8                    pypi_0    pypi
jupyterlab_pygments       0.1.2                      py_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
jupyterlab_widgets        1.0.0              pyhd3eb1b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
kiwisolver                1.4.4                    pypi_0    pypi
knn-pytorch               0.1                      pypi_0    pypi
lame                      3.100                h7b6447c_0    defaults
lcms2                     2.12                 h3be6417_0    defaults
ld_impl_linux-64          2.38                 h1181459_1    defaults
lerc                      3.0                  h295c915_0    defaults
libcublas                 11.10.3.66                    0    nvidia
libcufft                  10.7.2.124           h4fbf590_0    nvidia
libcufile                 1.7.0.149                     0    nvidia
libcurand                 10.3.3.53                     0    nvidia
libcusolver               11.4.0.1                      0    nvidia
libcusparse               11.7.4.91                     0    nvidia
libdeflate                1.17                 h5eee18b_0    defaults
libffi                    3.4.4                h6a678d5_0    defaults
libgcc-ng                 11.2.0               h1234567_1    defaults
libgfortran-ng            8.2.0                hdf63c60_1    anaconda
libgomp                   11.2.0               h1234567_1    defaults
libiconv                  1.16                 h7f8727e_2    defaults
libidn2                   2.3.4                h5eee18b_0    defaults
libnpp                    11.7.4.75                     0    nvidia
libnvjpeg                 11.8.0.2                      0    nvidia
libopenblas               0.3.2                h9ac9557_1    anaconda
libpng                    1.6.39               h5eee18b_0    defaults
libsodium                 1.0.18               h7b6447c_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libstdcxx-ng              11.2.0               h1234567_1    defaults
libtasn1                  4.19.0               h5eee18b_0    defaults
libtiff                   4.5.0                h6a678d5_2    defaults
libunistring              0.9.10               h27cfd23_0    defaults
libwebp                   1.2.4                h11a3e52_1    defaults
libwebp-base              1.2.4                h5eee18b_1    defaults
lz4-c                     1.9.4                h6a678d5_0    defaults
markdown                  3.4.3                    pypi_0    pypi
markupsafe                2.1.3                    pypi_0    pypi
matplotlib                3.5.3                    pypi_0    pypi
matplotlib-inline         0.1.6            py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
minkowskiengine           0.5.4                    pypi_0    pypi
mistune                   3.0.1                    pypi_0    pypi
mkl                       2021.4.0           h06a4308_640    defaults
mkl-service               2.4.0            py37h7f8727e_0    defaults
mkl_fft                   1.3.1            py37hd3c417c_0    defaults
mkl_random                1.2.2            py37h51133e4_0    defaults
multiprocess              0.70.14                  pypi_0    pypi
nbclassic                 1.0.0                    pypi_0    pypi
nbclient                  0.7.4                    pypi_0    pypi
nbconvert                 7.6.0                    pypi_0    pypi
nbformat                  5.8.0                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0    defaults
nest-asyncio              1.5.6            py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
nettle                    3.7.3                hbbd107a_1    defaults
networkx                  2.6.3                    pypi_0    pypi
ninja                     1.11.1                   pypi_0    pypi
notebook                  6.5.4                    pypi_0    pypi
notebook-shim             0.2.3                    pypi_0    pypi
numpy                     1.21.5           py37h6c91a56_3    defaults
numpy-base                1.21.5           py37ha15fc14_3    defaults
oauthlib                  3.2.2                    pypi_0    pypi
open3d                    0.9.0.0                  py37_0    open3d-admin
openblas-devel            0.3.2                         0    anaconda
opencv-python             4.8.0.74                 pypi_0    pypi
openh264                  2.1.1                h4ff587b_0    defaults
openssl                   1.1.1s               h7f8727e_0    anaconda
packaging                 23.1                     pypi_0    pypi
pandas                    1.3.5                    pypi_0    pypi
pandocfilters             1.5.0              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
parso                     0.8.3              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pathtools                 0.1.2                    pypi_0    pypi
pexpect                   4.8.0              pyhd3eb1b0_3    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pickleshare               0.7.5           pyhd3eb1b0_1003    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pillow                    9.4.0            py37h6a678d5_0    defaults
pip                       23.2.1                   pypi_0    pypi
pkgutil-resolve-name      1.3.10           py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pointnet2                 0.0.0                    pypi_0    pypi
pointnet2-cuda            0.0.0                    pypi_0    pypi
prometheus-client         0.17.0                   pypi_0    pypi
prometheus_client         0.14.1           py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
prompt-toolkit            3.0.39                   pypi_0    pypi
protobuf                  3.20.0                   pypi_0    pypi
psutil                    5.9.5                    pypi_0    pypi
ptyprocess                0.7.0              pyhd3eb1b0_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pyasn1                    0.5.0                    pypi_0    pypi
pyasn1-modules            0.3.0                    pypi_0    pypi
pycparser                 2.21               pyhd3eb1b0_0    defaults
pygments                  2.15.1                   pypi_0    pypi
pyopenssl                 23.0.0           py37h06a4308_0    defaults
pyparsing                 3.1.0                    pypi_0    pypi
pyrsistent                0.19.3                   pypi_0    pypi
pyserial                  3.5                      pypi_0    pypi
pysocks                   1.7.1                    py37_1    defaults
python                    3.7.16               h7a1cb2a_0    defaults
python-dateutil           2.8.2              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python-fastjsonschema     2.16.2           py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python-json-logger        2.0.7                    pypi_0    pypi
pytorch                   1.13.1          py3.7_cuda11.7_cudnn8.5.0_0    pytorch
pytorch-cuda              11.7                 h778d358_5    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2023.3                   pypi_0    pypi
pywavefront               1.3.3                    pypi_0    pypi
pywavelets                1.3.0                    pypi_0    pypi
pyyaml                    6.0                      pypi_0    pypi
pyzmq                     25.1.0                   pypi_0    pypi
readline                  8.2                  h5eee18b_0    defaults
requests                  2.28.1           py37h06a4308_0    defaults
requests-oauthlib         1.3.1                    pypi_0    pypi
rfc3339-validator         0.1.4                    pypi_0    pypi
rfc3986-validator         0.1.1                    pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
ruamel-yaml               0.17.32                  pypi_0    pypi
ruamel-yaml-clib          0.2.7                    pypi_0    pypi
scikit-image              0.19.3                   pypi_0    pypi
scikit-learn              1.0.2                    pypi_0    pypi
scipy                     1.7.3                    pypi_0    pypi
send2trash                1.8.2                    pypi_0    pypi
sentry-sdk                1.28.1                   pypi_0    pypi
setproctitle              1.3.2                    pypi_0    pypi
setuptools                68.0.0                   pypi_0    pypi
six                       1.16.0             pyhd3eb1b0_1    defaults
smmap                     5.0.0                    pypi_0    pypi
sniffio                   1.3.0                    pypi_0    pypi
soupsieve                 2.4.1                    pypi_0    pypi
sqlite                    3.41.2               h5eee18b_0    defaults
tensorboard               2.3.0                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
terminado                 0.17.1           py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
testpath                  0.6.0            py37h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
threadpoolctl             3.1.0                    pypi_0    pypi
tifffile                  2021.11.2                pypi_0    pypi
tinycss2                  1.2.1                    pypi_0    pypi
tk                        8.6.12               h1ccaba5_0    defaults
tomli                     2.0.1                    pypi_0    pypi
tomlkit                   0.11.8                   pypi_0    pypi
torchaudio                0.13.1               py37_cu117    pytorch
torchvision               0.14.1               py37_cu117    pytorch
tornado                   6.2              py37h5eee18b_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
tqdm                      4.65.0                   pypi_0    pypi
traitlets                 5.9.0                    pypi_0    pypi
transforms3d              0.3.1                    pypi_0    pypi
trimesh                   3.22.3                   pypi_0    pypi
typing_extensions         4.3.0            py37h06a4308_0    defaults
uri-template              1.3.0                    pypi_0    pypi
urllib3                   1.26.14          py37h06a4308_0    defaults
vision-sandbox            0.1                      pypi_0    pypi
wandb                     0.15.5                   pypi_0    pypi
wcwidth                   0.2.6                    pypi_0    pypi
webcolors                 1.13                     pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.6.1                    pypi_0    pypi
werkzeug                  2.2.3                    pypi_0    pypi
wheel                     0.41.2                   pypi_0    pypi
widgetsnbextension        4.0.8                    pypi_0    pypi
xz                        5.4.2                h5eee18b_0    defaults
y-py                      0.5.9                    pypi_0    pypi
ypy-websocket             0.8.2                    pypi_0    pypi
zeromq                    4.3.4                h2531618_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
zipp                      3.15.0                   pypi_0    pypi
zlib                      1.2.13               h5eee18b_0    defaults
zstd                      1.5.5                hc292b87_0    defaults
~~~

### 2. Configure KNN (K-nearest neighbors) accelerated using cuda programming:

In this paper, K-nearest neighbors (KNN) is primarily used to find the n nearest points within the grasp perception field for graspable point.
~~~shell
# cd YourProject/ and clone our knn code
git clone https://github.com/upc-ghy/knn.git
# cd KNN
cd KNN
# install
sudo python setup.py install

# you can test if it is successfully installed
# a simple case
python test.py
~~~ 

### 3. Configure MinkowskiEngine
~~~shell
# Enter the MinkowskiEngine official website, navigate to the section for installing CUDA 11.X using Anaconda
https://github.com/NVIDIA/MinkowskiEngine

conda create -n py3-mink python=3.8  # Skipped as we have already created it
conda activate py3-mink       # Here we activate our created environment 'conda activate grasp'

conda install openblas-devel -c anaconda  # Installing necessary packages
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia  # Not executed as already installed
# We have installed pytorch: conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install MinkowskiEngine

# Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
# export CUDA_HOME=/usr/local/cuda-11.1   This is already configured in the environment variables, our CUDA version is 11.7
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"  # This online method encountered numerous errors, resolution unknown; opting for the local method below

# Or if you want local MinkowskiEngine  # Installing locally using this method
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
~~~

### 4. Configure graspnetAPI (graspnetAPI is a tutorial document for data processing, pose visualization, and other functions of the GraspNet-1Billion dataset.)
~~~shell
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
~~~

### 5. Generation Tolerance Label
~~~shell
cd DataProcessing
sh command_generate_tolerance_label.sh
~~~

## Real Experiments
![platform](https://github.com/upc-ghy/GraspFast/assets/133858812/e0b68f88-1267-431d-962a-57b3d1faa2f5)


![Robot](https://github.com/upc-ghy/GraspFast/assets/133858812/b0ffb58a-b7c8-47e7-ae41-fb0ab08e83a5)


### Franka Grasp Experiment Environment

1. Ubuntu 20.04 (Our PC is installed with ubuntu system, you can choose other systems according to your needs.)

2. System real time kernel (Installing a real-time kernel is the key to implementing an operating system to drive Franka's real robots. It is recommended to install a real-time kernel with a similar version of the system's own kernel)

3. Nvidia Graphics Card Drivers (Graphics card drivers that meet the version of the real-time kernel system)

4. Ros Noetic (meet the Ubuntu 20.04)

5. Franka Robot (Install and compile libfranka, install and compile franka_ros. Remember that the version of libfranka is the same as the version of the Franka driver, and franka_ros is installed with the version corresponding to libfranka.)

6. MoveIt (Install moveit)
Start the franka robot arm correctly, activate the FCI and run franka_control：
~~~shell
roslaunch panda_moveit_config franka_control.launch robot_ip:=172.16.3.69
~~~

![Franka Control](https://github.com/upc-ghy/GraspBalance/raw/main/franka_control.png)


7. RealSense D435i (Install Intel RealSense Viewwe, and librealsense https://github.com/IntelRealSense/librealsense)

8. hand-eye calibration (Find the relationship between the camera coordinate system and the Franka robot coordinate system)
   
![Franka Control](https://github.com/upc-ghy/GraspBalance/raw/main/hand-eye_calibration.png)

10. Write Franka and RealSense control code (The algorithm predicts the grasping position in the camera coordinate system, converts it to a position in the Franka robot coordinate system, and controls the robot to move to that position for object grasping and placing into the storage box.)


# Contact information of some authors:

**Haiyuan Gui:** guihaiyuan736570@gmail.com

**Shanchen Pang:** pangsc@upc.edu.cn

**Xiao He:** xiao_he2022@126.com



