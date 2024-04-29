#!/home/nvidia/env/bin/python
import os
import cv2
import time
import torch
import numpy as np
import warnings

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from cv_bridge import CvBridge

from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Bool, Int32MultiArray, Int32
from sensor_msgs.msg import Image, CompressedImage
from ankle_band_tracking_interfaces.msg import BoundingBox, BoundingBoxArray
from ankle_band_tracking_interfaces.srv import ChooseTarget, ClearTarget

from .deep_sort import build_tracker
from .utils.draw import draw_boxes

class Tracker(Node):

    def __init__(self):
        super().__init__('tracker')

        # Declare params 
        # Deepsort params
        self.declare_parameter('max_dist', 0.2)
        self.declare_parameter('min_confidence', 0.3)
        self.declare_parameter('nms_max_overlap', 0.5)
        self.declare_parameter('max_iou_distance', 0.7)
        self.declare_parameter('max_age', 70)
        self.declare_parameter('n_init', 3)
        self.declare_parameter('nn_budget', 100)
        self.declare_parameter('distance', 'cosine')
        self.declare_parameter('reid_ckpt', '')

        # Detector params
        self.declare_parameter('classes', '')

        # Tracker params
        self.declare_parameter('camera_fov', 60)
        self.declare_parameter('save_path', '')

        # Misc params
        self.declare_parameter('frame_interval', 1)
        self.declare_parameter('save_results', False)
        self.declare_parameter('img_topic', '/image_raw')
        self.declare_parameter('bbox_topic', '/yolo_detection/detector/bounding_boxes')
        
        # Get params
        reid_ckpt = self.get_parameter('reid_ckpt').get_parameter_value().string_value
        classes_path = self.get_parameter('classes').get_parameter_value().string_value
        with open(classes_path, 'r', encoding='utf8') as fp:
            self.class_names = [line.strip() for line in fp.readlines()]
        
        save_path = self.get_parameter('save_path').get_parameter_value().string_value
        
        if not reid_ckpt or not save_path:
            raise Exception("Invalid or empty paths provided in YAML file.")
            
        deepsort_params = {
            "MAX_DIST": self.get_parameter('max_dist').get_parameter_value().double_value,
            "MIN_CONFIDENCE": self.get_parameter('min_confidence').get_parameter_value().double_value,
            "NMS_MAX_OVERLAP": self.get_parameter('nms_max_overlap').get_parameter_value().double_value,
            "MAX_IOU_DISTANCE": self.get_parameter('max_iou_distance').get_parameter_value().double_value,
            "MAX_AGE": self.get_parameter('max_age').get_parameter_value().integer_value,
            "N_INIT": self.get_parameter('n_init').get_parameter_value().integer_value,
            "NN_BUDGET": self.get_parameter('nn_budget').get_parameter_value().integer_value,
            "DISTANCE": self.get_parameter('distance').get_parameter_value().string_value,
            "REID_CKPT": reid_ckpt
        }

        args = {
            "frame_interval": self.get_parameter('frame_interval').get_parameter_value().integer_value,
            "save_results": self.get_parameter('save_results').get_parameter_value().bool_value,
            "img_topic": self.get_parameter('img_topic').get_parameter_value().string_value,
            "bbox_topic": self.get_parameter('bbox_topic').get_parameter_value().string_value
        }
        
        self.camera_fov = self.get_parameter("camera_fov").get_parameter_value().integer_value
        self.results_path = save_path

        # Subscribers
        if args["bbox_topic"] and args["img_topic"]:
            self.bbox_subscriber = self.create_subscription(BoundingBoxArray, args["bbox_topic"], self.ros_deepsort_callback, 1)
            self.img_subscriber = self.create_subscription(Image, args["img_topic"], self.img_callback, 1)
        else:
            raise Exception("no topic given. Ending node.")

        # Initialize publishers
        self.bbox_pub = self.create_publisher(Point, "/yolo_detection/tracker/bbox_center", 1)
        self.angle_pub = self.create_publisher(Float32, "/yolo_detection/tracker/target_angle", 1)

        self.target_present_pub = self.create_publisher(Bool, "/yolo_detection/tracker/target_present", 1)
        self.target_indice_pub = self.create_publisher(Int32, "/yolo_detection/tracker/target_indice", 1)

        self.detections_pub = self.create_publisher(Int32MultiArray, "/yolo_detection/tracker/detection_indices", 1)

        self.image_pub = self.create_publisher(Image, "/yolo_detection/tracker/deepsort_image", 1)

        # Initialize services to interact with node
        self.target_clear_srv = self.create_service(ClearTarget, "/yolo_detection/tracker/clear_target", self.clear_track_callback)
        self.target_choose_srv = self.create_service(ChooseTarget, "/yolo_detection/tracker/choose_target", self.choose_target_callback)

        self.cfg = {"DEEPSORT": deepsort_params}
        self.args = args

        self.logger = self.get_logger()
        self.cv_bridge = CvBridge()
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        # Build tracker
        self.deepsort = build_tracker(self.cfg, use_cuda=use_cuda)

        self.idx_frame = 0
        self.idx_tracked = None
        self.bbox_xyxy = []
        self.identities = []
        self.img_msg = None

        self.writer = None

        self.logger.info('Tracker node has started')


    def __del__(self):
        if self.writer:
            self.writer.release()

    def clear_track_callback(self, request, response):
        if self.idx_tracked is not None and request.clear:
            self.idx_tracked = None
            response.success = True
        else:
            response.success = False
        return response

    def choose_target_callback(self, ros_data, response):
        if self.idx_tracked is None:
            for identity in self.identities:
                if identity == ros_data.target:
                    self.idx_tracked = ros_data.target
                    response.success = True
                    return response
            
            response.success = False
            return response
        else:
            response.success = False
            return response
        return response

    def img_callback(self, msg):
        if msg:
            self.img_msg = msg

    # Main deepsort callback function
    def ros_deepsort_callback(self, msg):
        
        start = time.time()

        if self.img_msg == None:
            return

        # Convert ros Image message to opencv
        ori_im = self.cv_bridge.imgmsg_to_cv2(self.img_msg, "bgr8")  
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

        # Skip frame per frame interval
        self.idx_frame += 1
        if self.idx_frame % self.args["frame_interval"]:
            return
            
        # Parse ros message
        bbox_xywh = []
        cls_conf = []
        cls_ids = []
        for bounding_box in msg.bounding_boxes:
            x = bounding_box.xmin
            y = bounding_box.ymin
            w = bounding_box.xmax - bounding_box.xmin
            h = bounding_box.ymax - bounding_box.ymin

            bbox_center_x = float(x + w // 2)
            bbox_center_y = float(y + h // 2)
            bbox_size_x = float(w)
            bbox_size_y = float(h)

            bbox_xywh.append([bbox_center_x, bbox_center_y, bbox_size_x, bbox_size_y])
            cls_conf.append(bounding_box.conf)
            cls_ids.append(bounding_box.class_idx)

        bbox_xywh = np.array(bbox_xywh)
        cls_conf = np.array(cls_conf)
        cls_ids = np.array(cls_ids)
        
        # Select target class
        mask = cls_ids==0
        cls_conf = cls_conf[mask]
        bbox_xywh = bbox_xywh[mask]
        # Bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
        bbox_xywh[:,3:] *= 1.2 
        
        # Do tracking
        outputs = self.deepsort.update(bbox_xywh, cls_conf, im, tracking_target=self.idx_tracked)

        # If detection present draw bounding boxes
        if len(outputs) > 0:
            bbox_tlwh = []
            self.bbox_xyxy = outputs[:,:4]
            # detection indices
            self.identities = outputs[:,-1]
            ori_im = draw_boxes(ori_im, self.bbox_xyxy, self.identities)

            for bb_xyxy in self.bbox_xyxy:
                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

        end = time.time()

        # Draw frame count
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        frame_count = ("Frame no: %d" % self.idx_frame)
        cv2.putText(ori_im,frame_count, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        # Draw tracking number
        if self.idx_tracked:
            tracking_str = ("Tracking: %d" % self.idx_tracked)
        else:
            tracking_str = ("Tracking: None")

        bottomLeftCornerOfText = (10,550)
        cv2.putText(ori_im,tracking_str, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
        # Publish new image
        self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(ori_im, "bgr8"))

        if self.args["save_results"]:
            if not self.writer:
                os.makedirs(self.results_path, exist_ok=True)

                # path of saved video and results
                save_video_path = os.path.join(self.results_path, "results.avi")

                # create video writer
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.writer = cv2.VideoWriter(save_video_path, fourcc, 20, tuple(ori_im.shape[1::-1]))
            
            self.writer.write(ori_im)

        # Logging
        self.logger.info("frame: {}, time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                        .format(self.idx_frame, end-start, 1/(end-start), bbox_xywh.shape[0], len(outputs)))
    
        """publishing to topics"""
        # Publish detection identities
        self.detections_pub.publish(Int32MultiArray(data=self.identities))
        
        # Publish if target present
        if len(outputs) == 0 or self.idx_tracked is None:
            self.target_present_pub.publish(Bool(data=False))
        elif len(outputs) > 0 and self.idx_tracked:
            self.target_present_pub.publish(Bool(data=True))

        # Publish angle and xy data
        if self.idx_tracked is not None:

            x_center = (self.bbox_xyxy[0][0] + self.bbox_xyxy[0][2])/2
            y_center = (self.bbox_xyxy[0][1] + self.bbox_xyxy[0][3])/2

            pixel_per_angle = im.shape[1]/self.camera_fov

            x_center_adjusted = x_center - (im.shape[1]/2)

            angle = x_center_adjusted/pixel_per_angle

            self.bbox_pub.publish(Point(x=float(x_center), y=float(y_center), z=0.0))
            self.angle_pub.publish(Float32(data=angle))
            self.target_indice_pub.publish(Int32(data=self.idx_tracked))
        else:
            self.target_indice_pub.publish(Int32(data=-1))
            self.bbox_pub.publish(Point(x=0.0, y=0.0, z=0.0))


def main(args=None):
    rclpy.init(args=args)
    tracking_node = Tracker()
    try:
        rclpy.spin(tracking_node)
    except KeyboardInterrupt:
        pass
    tracking_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
