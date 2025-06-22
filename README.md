# Project in experts in teams
As part of a project robotics engineers and software engineers were tasked with solving an "industrial challange".
The use case given in the challenge is posed by Danfoss Drives, who manufacture AC drives for industrial use.
Individual components of the drives can be switched out depending on the needed specifications, which means that there are theoretically millions of combinations. 
The assembly is a series of snap-fits and screwing operations. 
The issue is that tolerances tend to stack up until they are outside the tolerance limits of the robot program, causing the operation to fail.
The challenge is to trace misalignment and make screwing operations more robust by compensating for them in screwing operations.

# Our solution
We used closed-loop eye-in-hand visual servoing to locate a hole and insert a screw.
An intel realsense camera was mounted to the end effector of a KUKA LBR IIWA 14 robot, along with a stick as a stand-in for a skrewdriver.
![ee_and_cam_mount](https://github.com/user-attachments/assets/426a1b72-31da-41f6-a025-4109b2cee552)

Because of the screwing tool placed at the center of the end effector, the placement of the camera is offset from the center. 
To compensate, the camera is tilted such that it points at the end of the tool, to maximize the view.
To unwarp the perspective we use four point-correspondences to estimate a homography between the camera plane and the plane containing the screw hole. 
The points we use for the homography are defined in the base frame of the robot.
They are the corners of a square centered on the end effector and at the height of the screw hole.
These points are then projected into the camera view, resulting in the four points seen here:

![camera_view](https://github.com/user-attachments/assets/eacc21f8-f80e-47e6-b5b5-51e64b7e5b56)

To detect screw holes, we use the hough transform method to extract circular features.
It operates on a binary image of image edges.
To generate the edge map we’re using the Canny edge detector.

![CV_system_demo](https://github.com/user-attachments/assets/df58b88d-8454-4ad2-9b59-e06df27aa506)

Since the the homography is centered around the end effector, tracking a specific hole in the image is to keep it centered in the image.
The image plane error can be defined as the vector from the circle center coordinate to the image center coordinate.
The transform from the robot base to the image plane is available, and can be used to obtain the image plane error in the robot base frame.
A proportional controller ensures that the Cartesian velocity of the end-effector is proportional to the image error in the base frame.
