import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np
import cv2 as cv
from skimage.feature import canny
PATH_TO_NODE = '/'.join(get_package_prefix('o3dOcv').split('/')[:-2])

from rcl_interfaces.msg import SetParametersResult

class Param():
    canny_blur_kernel_size = 3
    canny_blur_kernel_size = 3
    canny_threshold_low = 0.6
    canny_theshold_high = 0.95
    hough_accumulator_threshold = 30
    hough_accumulator_to_img_resolution = 1.0
    hough_min_dist_between_circles = 200
    hough_min_circle_size = 0
    hough_max_circle_size = 100
    hough_canny_threshold = 100
    
    @staticmethod
    def declare(node):
        class_variables = [(key, getattr(Param, key)) for key in set(vars(Param)) - set(['declare', 'on_params_changed', '__module__', '__dict__', '__init__', '__weakref__', '__doc__'])]
        node.declare_parameters(namespace='', parameters=class_variables)
        setattr(node, 'on_params_changed', Param.on_params_changed)
        node.add_on_set_parameters_callback(node.on_params_changed)
    
    @staticmethod
    def on_params_changed(params):
        for param in params:
            setattr(Param, param.name, param.value)
        return SetParametersResult(successful=True, reason='Parameter set')

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            1)
        self.edge_publisher = self.create_publisher(Image, 'camera_edge_images', 10)
        self.circle_image_publisher = self.create_publisher(Image, 'camera_circle_images', 10)
        self.img_converter = CvBridge()
        self.image_count = 0
        Param.declare(self)
        self.get_logger().info('Node has finished initialization')

    def listener_callback(self, msg):
        self.image_count += 1
        self.get_logger().info(f'I heard an image message {self.image_count}')
        img = self.img_converter.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        try:
            h = np.load(PATH_TO_NODE + '/homography_calibration.npy')
        except:
            self.get_logger().error("calibration file not found: run the python script calibration.py before running this node")
            exit()
        img = cv.warpPerspective(img, h, (90*4, 144*4))

        cv.imwrite("temp_image.png", img)
        img = ski.io.imread("temp_image.png")
        gray_img = ski.color.rgb2gray(img)

        edges = canny(gray_img, mode='reflect', 
                      sigma=Param.canny_blur_kernel_size, 
                      use_quantiles=True, 
                      low_threshold=Param.canny_threshold_low, 
                      high_threshold=Param.canny_theshold_high
                      )
        edges = ski.util.img_as_uint(edges)
        edges = cv.normalize(src=edges, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        circles = cv.HoughCircles(
            edges, 
            cv.HOUGH_GRADIENT, 
            Param.hough_accumulator_to_img_resolution, 
            Param.hough_min_dist_between_circles, 
            Param.hough_canny_threshold, 
            Param.hough_accumulator_threshold, 
            Param.hough_min_circle_size, 
            Param.hough_max_circle_size
            )

        if circles is not None:
            circles = np.uint(np.around(circles))
            for i in circles[0, :]:
                cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

        self.circle_image_publisher.publish(self.img_converter.cv2_to_imgmsg(img, encoding="passthrough"))
        self.edge_publisher.publish(self.img_converter.cv2_to_imgmsg(edges, encoding="passthrough"))

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()