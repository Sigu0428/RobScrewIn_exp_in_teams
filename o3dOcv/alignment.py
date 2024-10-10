import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix
from tf2_ros import TransformBroadcaster
import numpy as np
import cv2 as cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
#box dimensions for ros bag playback 90x144x50

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            1)
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
        img = self.img_converter.imgmsg_to_cv2(msg, desired_encoding='bgra8')
        cv2.imwrite("./images/image" + str(self.image_count) + ".jpg", img)
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        #gray_img = cv2.medianBlur(gray_img, 9)
        #circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=0, maxRadius=100)
        #circles = np.uint(np.around(circles))
        #for i in circles[0, :]:
        #    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        #cv2.imshow('image', img)
        #cv2.waitKey()

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()