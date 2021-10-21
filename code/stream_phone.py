#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


import cv2
import urllib
import numpy as np
# Publisher
image_pub = rospy.Publisher("side_camera", Image , queue_size=10)
bridge = CvBridge()
rospy.init_node('stream_phone', anonymous=True)
# Streaming
stream=urllib.urlopen('http://localhost:8080/videofeed')
bytes=''
while True:
	bytes += stream.read(16384)
	a = bytes.find('\xff\xd8')
	b = bytes.find('\xff\xd9')
	if a!=-1 and b!=-1:
		jpg = bytes[a:b+2]
		bytes= bytes[b+2:]
		cv_image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
		image_pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		print('Streaming!')
		print('----------')		
		#cv2.imshow('image',cv_image)
		#if cv2.waitKey(1) == 1:
		#	cv2.destroyWindow('image')
rospy.spin()