#!/home/nvidia/env/bin/python
import cv2
import torch
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2D
from ankle_band_tracking_interfaces.msg import BoundingBox, BoundingBoxArray

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Detector(Node):

    def __init__(self):
        super().__init__('detector')

        # Declare params
        # Yolo params
        self.declare_parameter("yolo_model.weight_file", '', ParameterDescriptor(description="Weights file"))
        self.declare_parameter("yolo_model.conf_thresh", 0.5, ParameterDescriptor(description="Confidence threshold"))
        self.declare_parameter("yolo_model.iou_thresh", 0.45, ParameterDescriptor(description="IOU threshold for NMS"))
        self.declare_parameter("yolo_model.device", '', ParameterDescriptor(description="Name of the device"))
        self.declare_parameter("yolo_model.img_size", 640, ParameterDescriptor(description="Image size"))
        self.declare_parameter("yolo_model.trace", False, ParameterDescriptor(description="Is traced model"))

        # Ros params
        self.declare_parameter('img_sub_topic', "/image_raw", ParameterDescriptor(description="Subscribed image topic"))
        self.declare_parameter('img_pub_topic', "/yolo_detection/detector/image", ParameterDescriptor(description="Image topic to publish"))
        self.declare_parameter('bbox_pub_topic', "/yolo_detection/detector/bounding_boxes", ParameterDescriptor(description="Bbox message to publish"))

        # Get params
        self.weights = self.get_parameter(name="yolo_model.weight_file").get_parameter_value().string_value
        self.conf_thresh = self.get_parameter(name="yolo_model.conf_thresh").get_parameter_value().double_value
        self.iou_thresh = self.get_parameter(name="yolo_model.iou_thresh").get_parameter_value().double_value
        self.device = self.get_parameter(name="yolo_model.device").get_parameter_value().string_value
        self.img_size = self.get_parameter(name="yolo_model.img_size").get_parameter_value().integer_value
        self.trace = self.get_parameter(name="yolo_model.trace").get_parameter_value().bool_value

        ros_params = {
            "img_sub_topic": self.get_parameter(name="img_sub_topic").get_parameter_value().string_value,
            "img_pub_topic": self.get_parameter(name="img_pub_topic").get_parameter_value().string_value,
            "bbox_pub_topic": self.get_parameter(name="bbox_pub_topic").get_parameter_value().string_value,
        }

        if not self.weights:
            raise Exception("Invalid or empty paths provided in YAML file.")
        if not ros_params["img_sub_topic"]:
            raise Exception("Invalid or empty paths provided in YAML file.")

        self.args = ros_params

        # Subscribers
        self.img_sub = self.create_subscription(Image, ros_params["img_sub_topic"], self.img_callback, 10)

        # Initialize publishers
        self.img_pub = self.create_publisher(Image, ros_params["img_pub_topic"], 10)
        self.bboxes_pub = self.create_publisher(BoundingBoxArray, ros_params["bbox_pub_topic"], 10)

        # Frame
        self.frame = None

        # Flags
        self.camera = False

        # Timer callback
        self.frequency = 20  # Hz
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)
    
        # Initialize yolov7
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device) # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size)
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

        # ROS attributes
        self.cv_bridge = CvBridge()

        self.get_logger().info("Detector Node has been started.")

    
    def img_callback(self, msg):
        if msg:
            self.get_logger().info("Camera Subscription is success.")
            self.frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.camera = True

    def timer_callback(self):
        if self.camera == True:
            self.yolov7_detection()

    def generate_bbox_msg(self, xyxy, conf, cls):
        bbox_msg = BoundingBox()

        bbox_msg.conf = conf.item()
        bbox_msg.xmin = int(xyxy[0].item())
        bbox_msg.ymin = int(xyxy[1].item())
        bbox_msg.xmax = int(xyxy[2].item())
        bbox_msg.ymax = int(xyxy[3].item())
        bbox_msg.class_idx = int(cls)
        bbox_msg.class_name = self.names[int(cls)]

        return bbox_msg


    def yolov7_detection(self):
        """ Perform object detection with custom yolov7-tiny"""

        # Flip image
        #img = cv2.flip(cv2.flip(np.asanyarray(self.frame),0),1) # Camera is upside down on the Go1
        img = self.frame

        im0 = img.copy()
        img = img[np.newaxis, :, :, :]
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)
        t3 = time_synchronized()

        # Process detections   
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                bboxes_array = BoundingBoxArray()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                    bboxes_array.bounding_boxes.append(self.generate_bbox_msg(xyxy, conf, cls))
                self.bboxes_pub.publish(bboxes_array)

                img_msg = self.cv_bridge.cv2_to_imgmsg(im0, "bgr8")
                img_msg.header.frame_id = "camera"
                self.img_pub.publish(img_msg)

            cv2.imshow("YOLOv7", im0) # cv2.resize(im0, None, fx=1.5, fy=1.5)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main(args=None):
    rclpy.init(args=args)
    detection_node = Detector()
    try:
        rclpy.spin(detection_node)
    except KeyboardInterrupt:
        pass
    detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
