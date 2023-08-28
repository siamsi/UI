#!/usr/bin/env python
# coding: utf-8


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import pandas as pd
import random
import time
import shutil
import cv2
import tqdm
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import *

import gradio as gr
import cv2 as cv
import os
from PIL import Image
from io import BytesIO
import requests
import base64
import pybase64


de=tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(de[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights

ip= ...
port= ...
path_cars= r'./data_test_segmentPlate'
url_inference= f'{ip}:{port}/plate_ocr'
def get_selected_image(evt:gr.SelectData):
    image_path= os.path.join(path_cars, os.listdir(path_cars)[evt.index])
    results=detect_image(yolo, image_path, input_size=YOLO_INPUT_SIZE, 
                         show=False, CLASSES=TRAIN_CLASSES,
                         rectangle_colors=(255,0,0))
    
    if len(results)==0:
        inferenced= 'پلاکی در تصویر یافت نشد'
        img= np.zeros((1, 1, 3))
    else:
        cv.imwrite('x.jpg', results[0])
        with open('x.jpg', "rb") as imageFile:
            img_base64 = pybase64.b64encode(imageFile.read())
        img_base64_to_str = img_base64.decode('utf-8')
        
        data = {
            'req_id': image_path,
            'vehicle_1': img_base64_to_str
        }
        
        response = requests.get(url_inference, json=data)
        inferenced= response.json()['ocr_result']
        img= cv.cvtColor(results[0], cv.COLOR_BGR2RGB)
        
    return inferenced, img

title= 'Our system title'
with gr.Blocks(theme='HaleyCH/HaleyCH_Theme', title= title) as demo:
# with gr.Blocks(title= title) as demo:

    with gr.Row():
        with gr.Column():
            pass
        with gr.Column():
            txt= gr.Textbox(label=title, scale=2)
        with gr.Column():
            pass
    list_images= [os.path.join(path_cars, file) for file in os.listdir(path_cars)]
    glry= gr.Gallery(value=list_images).style(grid=4, height='auto', selectable=True)
    
    txt= gr.Textbox(label="پیش بینی مدل")
    seg = gr.outputs.Image(type='numpy', label='پلاک سگمنت شده').style(full_width=False, height=180)
    glry.select(get_selected_image,outputs=[txt, seg])
        
demo.queue().launch()


# In[ ]:




