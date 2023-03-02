#!/usr/bin/env python3

import sys
sys.path.insert(1, "/home/avalocal/catkin_ws/src/laneatt_ros")
sys.path.insert(1, "/home/avalocal/anaconda3/envs/laneatt/lib/python3.8/site-packages")
sys.path.insert(1, "/media/avalocal/Samsung_T5/par/mmdetection3d")

import rospy
#import rospkg
#import time
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
import torch
#from sensor_msgs.msg import Image, CameraInfo
#import mmdet
#import cv2
#import pandas as pd
#from PIL import Image as im
#import matplotlib.pyplot as plt
import std_msgs.msg
#import message_filters
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.core.points import get_points_type
from geometry_msgs.msg import Quaternion
from scipy.spatial.transform import Rotation as R
import ros_numpy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

lim_x=[0, 50]
lim_y=[-25,25]
lim_z=[-3,3]

points_class = get_points_type('LIDAR')

#device = torch.device('cpu' if no_cuda else 'cuda:{}'.format("0"))

class trackclass():
    def __init__(self, model, poincloud_topic):
        torch.set_num_threads(4)
        self.poincloud_topic=poincloud_topic
        self.pcdSub=rospy.Subscriber(self.poincloud_topic, PointCloud2, self.read_pointcloud) #rospy.subscriber
        self.model=model
        self.bbox_publish = rospy.Publisher("pp_bboxes", BoundingBoxArray, queue_size=1)
        self.centerPub=rospy.Publisher("/centerPintCloud", PointCloud2, queue_size=1)
        #self.imgSub=message_filters.Subscriber("/image_proc_resize/camera_info", CameraInfo)
        self.fields=pc2.PointField(name='y', offset=4,datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='z', offset=8,datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='intensity', offset=12,datatype=pc2.PointField.FLOAT32, count=1)
        self.header = std_msgs.msg.Header()
        self.header.frame_id = 'velo_link'
        print('here')
        pointcloud=[]
        #self.vis=True
        #print("read pcd from rosbag")

        rospy.spin()

    def crop_pointcloud(self, pointcloud):
        # remove points outside of detection cube defined in 'configs.lim_*'
        mask = np.where((pointcloud[:, 0] >= 3) & (pointcloud[:, 0] <= 50) & (pointcloud[:, 1] >= -15) & (pointcloud[:, 1] <= 15) & (pointcloud[:, 2] >= -3) & (pointcloud[:, 2] <= 3))
        pointcloud = pointcloud[mask]
        return pointcloud

    def read_pointcloud(self,msgLidar):
        
        start=rospy.Time.now().to_sec()
        #read point cloud with numpify   #this one is faster
        pc = ros_numpy.numpify(msgLidar)
        
        points=np.zeros((pc.shape[0],4))
        points[:,0]=pc['x']
        points[:,1]=pc['y']
        points[:,2]=pc['z']
        #points[:,3]=pc['intensity']

        pc_arr=self.crop_pointcloud((points)) #to reduce computational expense

        pointcloud_np = points_class(pc_arr, points_dim=pc_arr.shape[-1], attribute_dims=None)
        
        result, _  = inference_detector(self.model, pointcloud_np)

        box = result[0]['boxes_3d'].tensor.numpy()
        scores = result[0]['scores_3d'].numpy()
        label = result[0]['labels_3d'].numpy()
 
        bbox_array=BoundingBoxArray()
        if len(box)!=0:
            #print("ls is ; ", ls)
            #ls=np.array(ls)
            inp=box[:,:3]
            print(inp)
            self.create_cloud(inp)

        for idx, box in enumerate(box):

            #print(box)
            bbox = BoundingBox()
            bbox.header.stamp = msgLidar.header.stamp
            bbox.header.frame_id = msgLidar.header.frame_id
            bbox.pose.position.z = box[2]
            bbox.pose.position.x = box[0]
            bbox.pose.position.y = box[1]
            bbox.pose.orientation.w = box[6]
            bbox.dimensions.x =box[3]
            bbox.dimensions.y = box[4]
            bbox.dimensions.z = box[5]
            bbox.value=scores[idx]
            bbox.label=label[idx]
            bbox_array.header = bbox.header
            bbox_array.boxes.append(bbox)

        bbox_array.header.frame_id = msgLidar.header.frame_id
        if len(bbox_array.boxes) != 0:
            self.bbox_publish.publish(bbox_array)
            bbox_array.boxes = []
        else:
            bbox_array.boxes = []
            self.bbox_publish.publish(bbox_array)

        end=rospy.Time.now().to_sec()
        print("time is:", end-start)  #0.0002
        #rospy.sleep(0.01)
        
    def yaw2quaternion(self, yaw) :
        return Quaternion(axis=[0,0,1], radians=yaw)
        
    def create_cloud(self, points):
        self.header.stamp=rospy.Time.now()
        self.centerpointcloud=pc2.create_cloud(self.header,self.fields, points)
        self.centerPub.publish(self.centerpointcloud)
        
    def pcd2bin(self, pointcloud):
        pass 

    def trackingFunction():
        pass
  
if __name__=='__main__':

    rospy.init_node("detectionNode")
    print("tracking node initialied")
    config_file = '/media/avalocal/Samsung_T5/par/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = '/media/avalocal/Samsung_T5/par/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
    #print("checkpoint is loaded")
    device= torch.device('cuda:0')
    topic="/kitti/velo/pointcloud"
    #with torch.no_grad():
    model = init_model(config_file, checkpoint_file, device)
    #print(model)
    #print("model is built")
    trackclass(model, topic)
    #rospy.spin()
  
