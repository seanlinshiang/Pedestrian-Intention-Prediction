# PedIntPredict
## Install

### Clone the repository recursively:

    git clone --recurse-submodules https://github.com/xavierlinxiang/PedIntPredict.git

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

### Build environments
    pip3 install -r requirements.txt


### Pytorch, Torchvision, Cudatoolkit

    conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch

### Install Gdown
    conda install -c conda-forge gdown

### DCNv2
    cd fair_mot/DCNv2/
    ./make.sh

### ffmpeg
    conda install -c conda-forge ffmpeg

### Download fairmot pretrained model:
    cd fair_mot/models
    gdown 1D70Oy8iahSOOMCXs6I0sum_ZHN8WBTV8


### Swig:

#### Installation of swig: 
    conda install -c anaconda swig
#### Build pose-estimation c++ library: 
    cd pose_estimation/tf_pose/pafprocess  
    swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

### Download graph path file:
    cd pose_estimation/models 
    gdown 1tOcNuXMQLCW0t0Kn4X_kWYtIFrluTO5u

