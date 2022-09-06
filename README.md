<div align="center">
  <br>
  <p>
    Face Detection is one of the most common and yet most complex deep learning problem. This model is made via a pipeline of Yolov5 and SSD using FaceNet (in Keras). Yolov5 is used for Face Detection in an image, video or live feed and the detected face is cropped and then tested via SSD to check whether it exists in the provided database.
  </p>

## <div align="center">Yolov5 Base Code</div>

This model is built upon a Yolov5 model that can be trained for a custom network. This model was paired wtih pre trained weight for Face detection in an image which was further optimized for the faces we required to work on. The original [Yolov5 repo](https://github.com/ultralytics/yolov5) can be used if any other network is to be built.

## <div align="center">Quick Start</div>
<div align="left">
<details open>
<summary>Install</summary>

Copy the required code in a directory. The requirements.txt file contains all the required packages in a [**Python>=3.7.0**](https://www.python.org/) environment, including [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/). 

In most case you'll be directly running the API, so no need to run the requirements.txt file.
</details>
<details open>
<summary>Run API</summary>

The best approach would be to create a conda enviornemnt and run the following command to automatically check the requirements and run the API.

```bash
cd files
python run.py
```

</details>
<details open>
<summary>Run detect.py</summary>

In case you directly want to use the console instead of the API, you can run the following commands.
**NOTE: YOU WILL HAVE TO RUN requirements.txt FIRST**
```bash
cd files
pip install -r requirements.txt  # install
```

Running detect.py directly gives you more control over the different arguments. Also, you can use different modes to run the inference.
For WebCam:
```bash
python detect.py --weights resources/face_yolo5.pt --conf 0.25 --source 0 --model_json resources/keras-facenet-h5/model.json --model_h5 resources/keras-facenet-h5/model.h5 --ssd_images resources
```

For Video Feed (Place video in resources/test_images and specify the name of file in --source given below):
```bash
python detect.py --weights resources/face_yolo5.pt --conf 0.25 --source resources/test_images/file_name.MOV --model_json resources/keras-facenet-h5/model.json --model_h5 resources/keras-facenet-h5/model.h5 --ssd_images resources
```

For Test Folder:
```bash
python model_api.py --weights resources/face_yolo5.pt --conf 0.25 --source resources/test_images --model_json resources/keras-facenet-h5/model.json --model_h5 resources/keras-facenet-h5/model.h5 --ssd_images resources
```

</details>








<details open>
<summary>Inference</summary>

YOLOv5 [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) inference. [Models](https://github.com/ultralytics/yolov5/tree/master/models) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>
</div>
