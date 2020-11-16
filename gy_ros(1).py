#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import roslib
import sys
import rospy
from rospy.numpy_msg import numpy_msg
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
#from cv_bridge import CvBridge
from xin import *
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

        image_sub = rospy.Subscriber("/pylon/raw_image_1",Image,self.main)
        rospy.spin()

    def main(self,data):


        frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

        image = Image2.fromarray(frame[..., ::-1])  # bgr to rgb

        # 这里直接调用我的函数，传入Image
        deal_img(image)


if __name__ == '__main__':
    Tracking('tracking_node')
