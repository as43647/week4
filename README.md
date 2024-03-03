# Object Detection

## Introduction

Mack a car counter using the [YOLO](https://docs.ultralytics.com/zh) algorithm.

If you done, push your code to the new repository and send me the link.

def checkDevice()
- 檢查裝置cuda是否支援

def checkVideo(videoPath)
- 檢查檢測影片是否輸入成功

def draw_boxes(img, className, pred, color=(255, 0, 255))
- 在影片圖像上繪製檢測到的目標物體(car)的邊界框框並回傳框的座標及confidence陣列及框的圖像

def main(videoPath, modelName)
- 載入模型、影片及mask和COCO dataset，先將mask resize成frame的大小並和frame疊合，並繪製計數的線，如果物件(car)通過偵測線，計數器將會加一

## Requirements

Create a conda environment with the required packages in your local machine.

```bash
conda create -n <myenv> python=3.8
conda activate <myenv>
# Install packages
conda install -r requirements.txt
```

### Download the data form [here](https://mailntustedutw-my.sharepoint.com/:f:/g/personal/m11107309_ms_ntust_edu_tw/Ek3a3ncMllBKjIQIuHkBYYMB9KR8E9MzU7a0niIOKmWWag?e=R6pyru)

- Video
- mask
- other

## Usage

```bash
conda activate <myenv>
python main.py
```

## References

- [Youtube](https://www.youtube.com/watch?v=WgPbbWmnXJ8)
- [Zone](https://www.computervision.zone/courses/object-detection-course/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
