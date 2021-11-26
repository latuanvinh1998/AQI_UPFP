# AQI using Periodic Frequent Pattern Mining

## I. Train Efficient parameters:
- --path: path to txt file, example in Data/Data_Train.txt
- --num_cls: number of class to classify, in this case is 6, because there are 6 level of PM25
- lr, epoch, cuda, batch_size: option setting
  ```sh
  python train_efficient.py --path Data/Data_Train.txt -- num_cls 6
  ```
## II. Create Transaction
- --path_img: path to image folder, example Data_Img/
- --ins_model: path to instance segmentation model, download [PointRed Model](https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl)
- --seg_model: path to instance segmentation model, download [here](https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.3/deeplabv3_xception65_ade20k.h5)
- --output_path: Folder to contain transaction file.
- --pm25: path to pm25 file. example Data/pm25.txt
- --haze: Use haze features or not, boolean type.
- --haze_file: path to haze file, example Data/haze.txt
```sh
python create_transaction.py --path_img Data_Img --ins_model pointrend_resnet50.pkl --seg_model deeplabv3_xception65_ade20k.h5 --out_path Data --pm25 Data/pm25.txt --haze True --haze_file Data/haze.txt
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
