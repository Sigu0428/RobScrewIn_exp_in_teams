import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2 as cv

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            1)
        self.img_converter = CvBridge()
        self.image_count = 0
        self.corners = []
        self.get_logger().info('Node has finished initialization')

    # Potential bug, the same object is getting transformed each iteration
    def listener_callback(self, msg):
        self.image_count += 1
        self.get_logger().info(f'I heard an image message {self.image_count}')
        self.img = self.img_converter.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        cv.imshow('left-click corners of the calibration square', self.img)
        cv.setMouseCallback('left-click corners of the calibration square', self.click_event, self)
        while(cv.waitKey() != 27):
            pass
        cv.destroyAllWindows()
        exit()
    
    @staticmethod
    def click_event(event, x, y, flags, self):
        if event == cv.EVENT_LBUTTONDOWN:
            self.img = cv.drawMarker(self.img, [x, y], [255, 0, 0])
            cv.imshow('left-click corners of the calibration square', self.img)
            self.corners.append([x, y])
        if len(self.corners) >= 4:
            img_pts = np.array(self.corners)
            calibr_pts = np.array([[0, 0], [0, 144*4], [90*4, 0], [90*4, 144*4]])
            print(np.array(self.img).shape)
            h, status = cv.findHomography(img_pts, calibr_pts)
            np.save("homography_calibration", h, allow_pickle=False)
            print("Calibration success")
            exit()
        
def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()