#!/usr/bin/env python3

import sys
sys.path.insert(1, "/home/avalocal/catkin_ws/src/laneatt_ros")
sys.path.insert(1, "/home/avalocal/anaconda3/envs/laneatt/lib/python3.8/site-packages")
#sys.path.insert(1, "/home/avalocal/Desktop/par/mmdetection3d")
sys.path.insert(1, "/media/avalocal/Samsung_T5/par/mmdetection3d")
import rospy
#import rospkg
#import time
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
import torch

import std_msgs.msg
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.core.points import get_points_type
from geometry_msgs.msg import Quaternion
from scipy.spatial.transform import Rotation as R
import ros_numpy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from autoware_msgs.msg import DetectedObjectArray, DetectedObject

lim_x=[-50, 75]
lim_y=[-25,25]
lim_z=[-5,5]

points_class = get_points_type('LIDAR')

#device = torch.device('cpu' if no_cuda else 'cuda:{}'.format("0"))

class trackclass():
    def __init__(self, model, poincloud_topic):
        torch.set_num_threads(4)
        self.poincloud_topic=poincloud_topic
        self.pcdSub=rospy.Subscriber(self.poincloud_topic, PointCloud2, self.read_pointcloud) #rospy.subscriber
        self.model=model
        
        pointcloud=[]
        #publish for autoware
        self.obj_publish_auto=rospy.Publisher('objDetection_autoware', DetectedObjectArray, queue_size=1)

        rospy.spin()

    def crop_pointcloud(self, pointcloud):
        # remove points outside of detection cube defined in 'configs.lim_*'
        mask = np.where((pointcloud[:, 0] >= lim_x[0]) & (pointcloud[:, 0] <= lim_x[1]) & (pointcloud[:, 1] >= lim_y[0]) & (pointcloud[:, 1] <= lim_y[1]) & (pointcloud[:, 2] >= lim_z[0]) & (pointcloud[:, 2] <= lim_z[1]))
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
        points[:,3]=1#pc['intensity']

        pc_arr=self.crop_pointcloud((points)) #to reduce computational expense

        pointcloud_np = points_class(pc_arr, points_dim=pc_arr.shape[-1], attribute_dims=None)
        
        result, _  = inference_detector(self.model, pointcloud_np)

        box = result[0]['boxes_3d'].tensor.numpy()
        #print(result)
        scores = result[0]['scores_3d'].numpy()
        label = result[0]['labels_3d'].numpy()
        label_definitions = ['person','bike','car']
        detected_object_msg=DetectedObjectArray()
        detected_object_msg.header.stamp = msgLidar.header.stamp
        detected_object_msg.header.frame_id = msgLidar.header.frame_id

        for idx, box in enumerate((box)):
            #print(box)
            detected_object = DetectedObject()
            detected_object.header.stamp = msgLidar.header.stamp
            detected_object.header.frame_id = msgLidar.header.frame_id
            detected_object.space_frame = msgLidar.header.frame_id
            detected_object.pose.position.z = box[2]
            detected_object.pose.position.x = box[0]
            detected_object.pose.position.y = box[1]
            detected_object.pose.orientation.w = box[6]
            detected_object.valid = True
            detected_object.pose_reliable = True
            detected_object.velocity_reliable = True
            detected_object.acceleration_reliable = True
            detected_object.dimensions.x =box[3]
            detected_object.dimensions.y = box[4]
            detected_object.dimensions.z = box[5]
            detected_object.score =scores[idx]
            detected_object.label=label_definitions[label[idx]]
            detected_object.header.frame_id= msgLidar.header.frame_id
     
           
            detected_object_msg.objects.append(detected_object)

        if len(detected_object_msg.objects) != 0:
            self.obj_publish_auto.publish(detected_object_msg)
            detected_object_msg.objects = []
        else:
            detected_object_msg.objects = []
            self.obj_publish_auto.publish(detected_object_msg)

        end=rospy.Time.now().to_sec()
        print("time is:", end-start)  #0.0002
        #rospy.sleep(0.01)
        
    def yaw2quaternion(self, yaw) :
        return Quaternion(axis=[0,0,1], radians=yaw)
        
    def create_cloud(self, points):
        self.header.stamp=rospy.Time.now()
        self.centerpointcloud=pc2.create_cloud(self.header,self.fields, points)
        self.centerPub.publish(self.centerpointcloud)
        
    
  
if __name__=='__main__':

    rospy.init_node("detectionNode")
    print("tracking node initialied")
    config_file = '/media/avalocal/Samsung_T5/par/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = '/media/avalocal/Samsung_T5/par/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
    #print("checkpoint is loaded")
    device= torch.device('cuda:0')
    topic="/lidar_tc/velodyne_points" #/lidar_tc/velodyne_points
    #with torch.no_grad():
    model = init_model(config_file, checkpoint_file, device)
    #print(model)
    #print("model is built")
    trackclass(model, topic)
    #rospy.spin()
  
