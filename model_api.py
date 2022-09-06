import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from PIL import Image as api_image
import streamlit as st
import torch.nn.functional as F

import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
from PIL import Image as conv_img
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
K.set_image_data_format('channels_last')

@smart_inference_mode()
def run(
        weights='resources/face_yolo5.pt',  # model.pt path(s)
        source='resources/test_images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.2,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model_json = 'resources/keras-facenet-h5/model.json',
        model_h5 = 'resources/keras-facenet-h5/model.h5',
        ssd_images = 'resources',
      
):
    start_time = time.time()

    st.set_page_config(
     page_title="FaceRecognizer",
     layout="wide",
    )

    st.header("Face Classification")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tab1, tab2, tab3, tab4= st.tabs(["Custom Image", "Webcam", "Test Folder Images", "Add Data"])

    api_time = time.time()

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    start_ssd_time = time.time()
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    my_model = model_from_json(loaded_model_json)
    my_model.load_weights(model_h5)
    end_ssd_time = time.time()

    tf.random.set_seed(1)

    FRmodel = my_model

    start_load_time = time.time()

    database1 = {}
    for images1 in os.listdir(ssd_images+"/images/db1"):
    
        # check if the image ends with jpg
        if (images1.endswith(".jpg")):
            database1[images1.replace('.jpg','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

        if (images1.endswith(".png")):
            database1[images1.replace('.png','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

        if (images1.endswith(".JPG")):
            database1[images1.replace('.JPG','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

        if (images1.endswith(".jpeg")):
            database1[images1.replace('.jpeg','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

    
    database2 = {}
    for images2 in os.listdir(ssd_images+"/images/db2"):
    
        # check if the image ends with jpg
        if (images2.endswith(".jpg")):
            database2[images2.replace('.jpg','')] = img_to_encoding(ssd_images+"/images/db2/" + images2, FRmodel)

        if (images2.endswith(".png")):
            database2[images2.replace('.png','')] = img_to_encoding(ssd_images+"/images/db2/" + images2, FRmodel)

        if (images2.endswith(".JPG")):
            database2[images2.replace('.JPG','')] = img_to_encoding(ssd_images+"/images/db2/" + images2, FRmodel)

        if (images2.endswith(".jpeg")):
            database2[images2.replace('.jpeg','')] = img_to_encoding(ssd_images+"/images/db2/" + images2, FRmodel)

    
    database3 = {}
    for images3 in os.listdir(ssd_images+"/images/db3"):
    
        # check if the image ends with jpg
        if (images3.endswith(".jpg")):
            database3[images3.replace('.jpg','')] = img_to_encoding(ssd_images+"/images/db3/" + images3, FRmodel)

        if (images3.endswith(".png")):
            database3[images3.replace('.png','')] = img_to_encoding(ssd_images+"/images/db3/" + images3, FRmodel)

        if (images3.endswith(".JPG")):
            database3[images3.replace('.JPG','')] = img_to_encoding(ssd_images+"/images/db3/" + images3, FRmodel)

        if (images3.endswith(".jpeg")):
            database3[images3.replace('.jpeg','')] = img_to_encoding(ssd_images+"/images/db3/" + images3, FRmodel)
    end_load_time = time.time()

    with st.sidebar:
        st.header("Initial Setup Details", )
        st.subheader("Loading times")
        st.text('API Load Time: {:4f} ms'.format(
                (api_time - start_time) * 1000))
        st.text('SSD model Load Time: {:4f} s'.format(
                (end_ssd_time - start_ssd_time)))
        st.text('SSD images Load Time: {:4f} s'.format(
                end_load_time - start_load_time))

    count = 0
    uploaded_file = tab1.file_uploader("Choose a image",type=['png','jpeg','jpg'])
    uploaded_file2 = tab4.file_uploader("Choose a Image")

    if tab1.button('Run Custom Inference'):
        since = time.time()
        count = 0
        if uploaded_file is not None:

            with open(os.path.join("resources/uploaded_images",uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer())    

                source = str('resources/uploaded_images/' + uploaded_file.name)
                save_img = not nosave and not source.endswith('.txt')  # save inference images
                is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
                is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
                webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
                if is_url and is_file:
                    source = check_file(source)  # download       

                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
                bs = 1  # batch_size

                # Run inference 
                model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
                seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                for path, im, im0s, vid_cap, s in dataset:
                    start_time = time.time()
                    with dt[0]:
                        im = torch.from_numpy(im).to(device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # Inference
                    with dt[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred = model(im, augment=augment, visualize=visualize)

                    # NMS
                    with dt[2]:
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        count = count + 1
                        seen += 1
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                crop = save_one_box(xyxy, imc, BGR=True, save = False)  
                                dist, name, exists = who_is_it(crop, database1, database2, database3, FRmodel)

                                if save_img or save_crop or view_img:  # Add box to image
                                    c = int(cls)  # integer class
                                    if (exists):
                                        label = None if hide_labels else (names[c] if hide_conf else f'{name} {dist:.2f}')
                                    else:
                                        label = None if hide_labels else (names[c] if hide_conf else f'Unidentified {dist:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                print("Inference time is:  %.6s seconds" % (time.time() - start_time))

                        # Stream results
                        im0 = annotator.result()
                        tab1.image(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))

                if update:
                    strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)          

            time_elapsed = time.time() - since
            tab1.text('=>   Inference Time: {:4f}'.format(
                    time_elapsed))
        else:
            tab1.text('Please upload an image')

    if tab3.button('Run Test Images'):
        since2 = time.time()
        source = str('resources/test_images')
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

        # Run inference 
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            start_time = time.time()
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                count = count + 1
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        crop = save_one_box(xyxy, imc, BGR=True, save = False)  
                        dist, name, exists = who_is_it(crop, database1, database2, database3, FRmodel)

                        if save_img or save_crop or view_img:  # Add box to image
                            c = int(cls)  # integer class
                            if (exists):
                                label = None if hide_labels else (names[c] if hide_conf else f'{name} {dist:.2f}')
                            else:
                                label = None if hide_labels else (names[c] if hide_conf else f'Unidentified {dist:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        print("Inference time is:  %.6s seconds" % (time.time() - start_time))

                # Stream results
                im0 = annotator.result()

                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        time_elapsed2 = time.time() - since2
        unit_time = time_elapsed2 / count
        tab3.text('=>   Inference Time: {:4f} s for a total of {:d} images'.format(
                time_elapsed2, count))
        tab3.text('=>  Approx unit inference time is: {:4f} s'.format(
                unit_time))

    run = tab2.checkbox('Run Webcam')
    FRAME_WINDOW = tab2.image([])
    camera = cv2.VideoCapture(0)
    while run:
        count = 0
        source = str(0)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference 
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            start_time = time.time()
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        crop = save_one_box(xyxy, imc, BGR=True, save = False)  
                        dist, name, exists = who_is_it(crop, database1, database2, database3, FRmodel)

                        if save_img or save_crop or view_img:  # Add box to image
                            c = int(cls)  # integer class
                            if (exists):
                                label = None if hide_labels else (names[c] if hide_conf else f'{name} {dist:.2f}')
                            else:
                                label = None if hide_labels else (names[c] if hide_conf else f'Unidentified {dist:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        print("Inference time is:  %.6s seconds" % (time.time() - start_time))

                # Stream results
                im0 = annotator.result()
                FRAME_WINDOW.image(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))

        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    else:
        tab2.write('Webcam Off')
        source = 'resources/test_images'

    user_input = tab4.text_input("Enter person name", "Unnamed")
    if tab4.button('Add Image'):
        since = time.time()
        count = 0
        if uploaded_file2 is not None:
            im = api_image.open(uploaded_file2)

            with open(os.path.join("resources/uploaded_images",user_input + ".jpg"),"wb") as f: 
                f.write(uploaded_file2.getbuffer())    

                source = str('resources/uploaded_images/' + user_input + ".jpg") 
                save_img = not nosave and not source.endswith('.txt')  # save inference images
                is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
                is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
                webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
                if is_url and is_file:
                    source = check_file(source)  # download       

                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
                bs = 1  # batch_size

                # Run inference 
                model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
                seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                for path, im, im0s, vid_cap, s in dataset:
                    start_time = time.time()
                    with dt[0]:
                        im = torch.from_numpy(im).to(device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # Inference
                    with dt[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred = model(im, augment=augment, visualize=visualize)

                    # NMS
                    with dt[2]:
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        count = count + 1
                        seen += 1
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                crop = save_one_box(xyxy, imc, BGR=True, save = False)  
                                cv2.imwrite('resources/images/db1/' + user_input + ".jpg", crop)
                                tab4.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                for images1 in os.listdir(ssd_images+"/images/db1"):
    
                                    # check if the image ends with jpg
                                    if (images1.endswith(".jpg")):
                                        database1[images1.replace('.jpg','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

                                    if (images1.endswith(".png")):
                                        database1[images1.replace('.png','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

                                    if (images1.endswith(".JPG")):
                                        database1[images1.replace('.JPG','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

                                    if (images1.endswith(".jpeg")):
                                        database1[images1.replace('.jpeg','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)


                if update:
                    strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)          

            time_elapsed = time.time() - since
            tab1.text('=>   Inference Time: {:4f}'.format(
                    time_elapsed))

        else:
            tab4.text('Please upload an image')


def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
    basic_loss = tf.maximum(tf.add(tf.subtract(pos_dist,neg_dist),alpha),0)
    loss = tf.reduce_sum(basic_loss)
    
    return loss

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

def pil_img_to_encoding(in_img, model):
    im = conv_img.fromarray(in_img)
    newsize = (160, 160)
    img = im.resize(newsize) 
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


def who_is_it(image_path, database1, database2, database3, model):
    encoding = pil_img_to_encoding(image_path,model)
    min_dist = 100
    for (name, db_enc) in database1.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
            if min_dist < 0.55:
                return min_dist, identity, True
    
    if min_dist > 0.6:
        for (name, db_enc) in database2.items():
            dist = np.linalg.norm(encoding - db_enc)
            if dist < min_dist:
                min_dist = dist
                identity = name
                if min_dist < 0.55:
                    return min_dist, identity, True

    if min_dist > 0.6:
        for (name, db_enc) in database3.items():
            dist = np.linalg.norm(encoding - db_enc)
            if dist < min_dist:
                min_dist = dist
                identity = name
                if min_dist < 0.55:
                    return min_dist, identity, True

    exists = False
        
    if min_dist > 0.90:
        print("Not in the database." + " Min dist is " + str(min_dist))
        exists = False
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        exists = True
        
    return min_dist, identity, exists

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='resources/face_yolo5.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='resources/test_images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--model_json', type=str, default='resources/keras-facenet-h5/model.json', help='Tell path for model')
    parser.add_argument('--model_h5', type=str, default='resources/keras-facenet-h5/model.h5', help='Tell path for model')
    parser.add_argument('--ssd_images', type=str, default='resources', help='Tell path for ssd Images')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
