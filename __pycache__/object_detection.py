# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:03:38 2018

@author: DELL
"""
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as  labelmap
from ssd import build_ssd
import imageio




def detect(video_frame,network,transform):
    frame_height,frame_width=video_frame.shape[:2]
    frame_trans=transform(video_frame)[0]
    x = torch.from_numpy(frame_trans).permute(2,0,1)
    x= Variable(x.unsqueeze(0))
    y= network(x)
    detections=y.data
    scale=torch.Tensor([frame_width,frame_height,frame_width,frame_height])
    for i in range(detections.size(1)):
        j=0
        while detections[ 0, i, j, 0] >=0.6:
            point = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(video_frame, (int(point[0]), int(point[1])), (int(point[2]), int(point[3])) ,(255,0,0),2)
            cv2.putText(frame, labelmap[i - 1], (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j=j+1
    return video_frame
    
network = build_ssd('test')
network.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location= lambda storage ,loc: storage))

transform = BaseTransform(network.size,(104/256.0,117/256.0, 123/256.0))

reader = imageio.get_reader('Bird2.MP4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('bird_output2.MP4',fps=fps)
for i,frame in enumerate(reader):
    frame=detect( frame ,network ,transform)
    writer.append_data(frame)
    print(i)
writer.close()
    

