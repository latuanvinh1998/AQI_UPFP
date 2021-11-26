# AQI using Periodic Frequent Pattern Mining

## I. Train Efficient parameters:
- --path: path to txt file, example in Data/Data_Train.txt
- --num_cls: number of class to classify, in this case is 6, because there are 6 level of PM25
- lr, epoch, cuda, batch_size: option setting
  ```sh
  sudo apt -y install cmake cmake-gui cmake-qt-gui
  sudo apt -y install qtbase5-dev
  sudo apt-get -y install qtdeclarative5-dev
  
  sudo apt -y install libcrypto++-dev 
  sudo apt -y install libgoogle-glog-dev 
  sudo apt -y install libgflags-dev
  sudo apt -y install --no-install-recommends libboost-all-dev
  ```
*If you are using Jetson you don't have to install Cuda/CuDNN/TensorRT.*

## II. Clone & Run
### Step 1: Clone project:
```sh
git clone https://github.com/CuteBoiz/JetsonNano-ArcFace
cd JetsonNano-ArcFace
```

### Step 2: Edit CmakeLists.txt

*If you are using Jetson Nano:* `rm CMakeLists.txt && mv CMakeLists.txt.jetson-nano  CMakeLists.txt`

```sh
gedit CMakeLists.txt 
```
- *Change my username (tanphatnguyen) to your username*
- *Change libtorch directory (line 10) (PC only)*
- *Change TensorRT version (line 17 & 18) (PC only)*

### Step 3: Compile & Run:

```sh
cmake .
make
./main
```

## References
- [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- [Arcface](https://arxiv.org/abs/1801.07698)
- [MobileFaceNets](https://arxiv.org/abs/1804.07573)
- [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

## To-Do
- [ ] Convert all engine to onnx.
- [ ] Add export TensorRT engine.
- [ ] Add Non GUI Inference code.
