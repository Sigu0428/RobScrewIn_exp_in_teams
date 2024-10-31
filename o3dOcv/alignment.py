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
        self.get_logger().info('Node has finished initialization')
        self.image_count = 0

    def broadcast_tf(self, T):
        t = TransformStamped()

        # header
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_depth_optical_frame'
        t.child_frame_id = 'skrew_position_estimate'

        # translation
        t.transform.translation.x = T[0, 3]
        t.transform.translation.y = T[1, 3]
        t.transform.translation.z = T[2, 3]

        # rotation
        rot = Rotation.from_matrix(T[0:3, 0:3])
        q = rot.as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    # Potential bug, the same object is getting transformed each iteration
    def listener_callback(self, msg):
        self.image_count += 1
        self.get_logger().info(f'I heard an image message {self.image_count}')
        img = self.img_converter.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        try:
            h = np.load('/'.join(get_package_prefix('o3dOcv').split('/')[:-2]) + '/homography_calibration.npy')
        except:
            self.get_logger().error("calibration file not found: run the python script calibration.py before running this node")
            exit()
        img = cv.warpPerspective(img, h, (90*4, 144*4))

        cv.imwrite("temp_image.png", img)
        img = ski.io.imread("temp_image.png")
        gray_img = ski.color.rgb2gray(img)

        edges = canny(gray_img, mode='reflect', sigma=3, use_quantiles=True, low_threshold=0.6, high_threshold=0.95)
        edges = ski.util.img_as_uint(edges)
        edges = cv.normalize(src=edges, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 200, param1=100, param2=30, minRadius=0, maxRadius=100)
        
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