#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import roslib
import sys
import rospy
from rospy.numpy_msg import numpy_msg
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
#from cv_bridge import CvBridge

import os
from timeit import time
import sys
import cv2
import numpy as np
from PIL import Image as Image2
import ctypes




class Tracking:
    def __init__(self,node_name):
        self.nname = node_name

        rospy.init_node(self.nname,anonymous=True)
        #self.bridge = CvBridge()
        #self.lib, self.encoder, self.metric, self.tracker = Init()

        #image_sub = rospy.Subscriber("/pylon/raw_image_1",Image,self.tracking_callback)
        image_sub = rospy.Subscriber("/pylon/raw_image_1",Image,self.main)
        rospy.spin()

    def main(self,data):


        frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        #print("data",type(frame),np.shape(frame))
        #print("frame",frame)
        
        #frame = frame[180:900, 320:1600]  # 把原始图中间抠图，抠图的结果大小为（720,1280）
        image = Image2.fromarray(frame[..., ::-1])  # bgr to rgb

        #这里直接调用我的函数，传入Image


if __name__ == '__main__':
    Tracking('tracking_node')
