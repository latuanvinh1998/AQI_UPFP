# AQI using Periodic Frequent Pattern Mining

## I. Train Efficient parameters:
- --path: path to txt file, example in Data/Data_Train.txt
- --num_cls: number of class to classify, in this case is 6, because there are 6 level of PM25
- lr, epoch, cuda, batch_size: option setting
  ```sh
  python train_efficient.py --path Data/Data_Train.txt -- num_cls 6
  ```
### Model will save in Model/model.pth

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
##### output contain data.csv is details for each image and data.txt for mining.

### III. Mining Data
- --input_file: Path to transaction file, example Data/data.txt.
- --output_file: Path to output file.
- --minSup: Minimum Support.
- --maxPer: Maximum Period.
  ```sh
  python mining.py --input_file Data/data.txt --output_file Data/result.txt --minSup 0.2 --maxPer 0.2
  ```
### IV. Inference:
 - --input_image: Path to input image, example Data_Img/20200228_121729A.jpg
 - --model_path: Trained model from train_efficient.py
 - --mined_pattern: Path to mined pattern file, example Data/result.txt or Data/PM.txt
 - --ins_model: path to instance segmentation model, download [PointRed Model](https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl)
 - --seg_model: path to instance segmentation model, download [here](https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.3/deeplabv3_xception65_ade20k.h5)

```sh
python inference.py input_image Data_Img/20200228_121729A.jpg --model_path Model/model.pth --mined_pattern Data/result.txt --ins_model pointrend_resnet50.pkl --seg_model deeplabv3_xception65_ade20k.h5
```

## V. Generate haze file:
  - Input folder is define as Data_Img and must be same location with haze.R, the output will be haze.txt and use for create_transaction.py \[--haze_file]
