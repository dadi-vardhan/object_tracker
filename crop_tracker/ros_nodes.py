#!../../../venv/bin/python3

import os 
import sys
from enum import Enum

# HOME = os.getcwd()
# sys.path.append(f"{HOME}/yolov7")

import numpy as np
import cv2
import torch.backends.cudnn as cudnn

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator

from tracker.byte_tracker import BYTETracker, STrack
from tracker_utils import box_iou_batch

from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_imshow


class StreamArgs(Enum):
    SOURCE: str = '/home/dadi_vardhan/Downloads/camera_0.mp4'# Path to video file
    FPS: int = 30 # Frames per second
    IMAGE_SIZE: int = 640 # Image size
    STRIDE: int = 32 # Stride
    

class VideoStreamer(Node):
    def __init__(self):
        super().__init__('video_streamer')

        self._pub_resized_img = self.create_publisher(
            msg_type = Image,
            topic = '/video_pub/img_resized',
            qos_profile = 10
        )

        self.dataset = self.prepare_dataloader()
        self.frames = iter(self.dataset)

        self._timer_rimg = self.create_timer(
            timer_period_sec = 1 / StreamArgs.FPS.value,
            callback = self._rimg_callback
        )


        self.bridge = CvBridge()
        
    def prepare_dataloader(self):
        webcam = StreamArgs.SOURCE.value.isnumeric() \
        or StreamArgs.SOURCE.value.endswith('.txt')  \
        or StreamArgs.SOURCE.value.lower().startswith((
            'rtsp://', 'rtmp://','http://', 'https://'
            ))
        
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(
                StreamArgs.SOURCE.value,
                img_size = StreamArgs.IMAGE_SIZE.value,
                stride = StreamArgs.STRIDE.value
                )
        else:
            dataset = LoadImages(
                path = StreamArgs.SOURCE.value,
                img_size = StreamArgs.IMAGE_SIZE.value,
                stride = StreamArgs.STRIDE.value
                )
        
        return dataset
        
    def _rimg_callback(self):
        path,img,im0s,_ = next(self.frames)

        msg = self.bridge.cv2_to_imgmsg(im0s, encoding='passthrough')
        
        self._pub_resized_img.publish(msg)
        #self.get_logger().info(f'Published frame {msg.header.frame_id}')


class VideoListener(Node):
    def __init__(self):
        super().__init__('video_listener')
        self._sub_image = self.create_subscription(
            msg_type = Image,
            topic = '/video_pub/img_resized',
            callback = self._sub_callback,
            qos_profile = 10
        )
        self.bridge = CvBridge()

    def _sub_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as error:
            self.get_logger().error(error)

        cv2.imshow("Received Image", cv_image)
        cv2.waitKey(1)


def main1(args=None):
    rclpy.init(args=args)
    image_publisher = VideoStreamer()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

# def main2(args=None):
#     rclpy.init(args=args)
#     image_subscriber = VideoListener()
#     rclpy.spin(image_subscriber)
#     image_subscriber.destroy_node()
#     rclpy.shutdown()

if __name__ == '__main__':
    main1()
    #main2()