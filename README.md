# PedIntPredict

## Repo contents
* `/fair_mot` - FairMOT module files
* `/PCPA_weights` - Weights for PCPA, both with and without speed branch 
* `/pose_estimation` - Skeleton fitting module files
* `/speed` - Speed file for PIE datasets
* `/speed_model` - Speed prediction model
* `/raft_core` - Speed prediction module
* `.gitignore` - Ignore misc files
* `.gitmodules` - Submodules for FairMOT
* `deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz` - Video segmentation model
* `action_predict.py` - Utils for action prediction
* `utils_PCPA.py` - Utils for PCPA
* `base_models.py` - Base models for PCPA
* `requirements.txt` - Requirements file for python
* `README.md` - Instructions on how to use this repo
* `main.py` - Main file for inferencing the video

Our implementation relied on 
1. https://github.com/mjpramirez/Volvo-DataX
2. https://github.com/OSU-Haolin/Pedestrian_Crossing_Intention_Prediction

## Install

### Clone the repository recursively:

    git clone --recurse-submodules https://github.com/xavierlinxiang/Pedestrian-Intention-Prediction.git

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

#### Download graph path file:
    cd pose_estimation/models 
    gdown 1tOcNuXMQLCW0t0Kn4X_kWYtIFrluTO5u

## Running the code
    python3 main.py 
        --source <path to the video file>
        --dense_model <path to the intention-prediction model>
        --speed_file <path to the speed file>
        --save-vid 
        --save-txt

### For more options or help
    python3 main.py -h



