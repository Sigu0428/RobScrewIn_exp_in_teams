import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix
from tf2_ros import TransformBroadcaster
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from tqdm import tqdm
import sklearn as skln
import scipy as scp
from scipy.spatial.transform import Rotation
import sensor_msgs_py.point_cloud2 as pc2
#box dimensions for ros bag playback 90x144x50

def preprocess_pcd(pcd, voxl_downsampl_rad=0.01, normals_k_nearest=100, featr_comp_rad=0.02):
    pcd = pcd.voxel_down_sample(voxl_downsampl_rad)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(normals_k_nearest))
    # shape = (N, 33)
    features = np.asarray(
    o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamRadius(featr_comp_rad)
    ).data.T
    )
    return pcd, features

def getAssetsPath():
    return '/'.join(get_package_prefix('o3dOcv').split('/')[:-2]) + '/assets/'

# This implementation of global_alignment is implmented by me (sigurd) for a different course and reused here
def global_alignment(scene, scene_features, object, object_features, kdtree, ransac_max_itr = 5000, ransac_inlier_threshold = 0.005, show_best=False, show_correspondences=False, show_result=False): # low noise freature_radius=0.005 # normals_k_nearest=10 ransac_threshold=0.0005

    # Match features between scene and object (euclidean distance in 33D feature space)
    correspondences = skln.metrics.pairwise_distances_argmin(object_features, scene_features) # f(X, Y) computes for each row in X the index of the row of Y which is closest
    correspondences = np.vstack((np.arange(object_features.shape[0]), correspondences)).T

    if show_correspondences:
        line_set = o3d.geometry.LineSet().create_from_point_cloud_correspondences(object, scene, o3d.utility.Vector2iVector(correspondences))
        o3d.visualization.draw_geometries([object, scene, line_set], window_name="correspondences")

    bestT = None
    bestT_inliers = 0
    pbar = tqdm(range(ransac_max_itr))
    for i in pbar:
        pbar.set_description("Global alignment")
        pbar.set_postfix({"best inliers": bestT_inliers})

        random_points = correspondences[np.random.choice(correspondences.shape[0], 3, replace=False), :]
        
        T = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(
            object, 
            scene, 
            o3d.utility.Vector2iVector(
                random_points
            )
        )
        candidate = o3d.geometry.PointCloud(object)
        candidate = candidate.transform(T)

        (d, idx) = kdtree.query(np.asarray(candidate.points), k=1, workers=-1) # distance is approximate within (1 + eps) times the real distance

        inliers = np.sum(d < ransac_inlier_threshold)
        if inliers >= bestT_inliers:
            bestT = T
            bestT_inliers = inliers
            if show_best:
                line_set = o3d.geometry.LineSet().create_from_point_cloud_correspondences(object, scene, o3d.utility.Vector2iVector(random_points))
                scene.paint_uniform_color([0.8, 0.8, 0.8])
                np.asarray(scene.colors)[idx[d < ransac_inlier_threshold], :] = [0, 0, 1]
                np.asarray(candidate.colors)[np.argwhere(d < ransac_inlier_threshold), :] = [0, 1, 0]
                o3d.visualization.draw_geometries([candidate, scene, line_set, object], window_name = 'inliers: ' + str(inliers))
                scene.paint_uniform_color([0.8, 0.8, 0.8])

    if show_result:
        object.paint_uniform_color([0.8, 0.2, 0.2])
        scene.paint_uniform_color([0.2, 0.2, 0.8])
        o3d.visualization.draw_geometries([object.transform(bestT), scene], window_name = 'Result of global alignment')

    return bestT

# This implementation of ICP is implemented by me (sigurd) for a different course and reused here
def ICP(scene, object_local, kdtree, max_itr = 1000, max_distance = 0.01, show_correspondences=False, show_result=False): #0.01
    object_local = o3d.geometry.PointCloud(object_local)
    scene = o3d.geometry.PointCloud(scene)
    
    n = len(object_local.points)
    
    T_accum = None
    pbar = tqdm(range(max_itr))
    for i in pbar:
        pbar.set_description("Iterative closest point")
        
        (d, iq) = kdtree.query(object_local.points, k=1, workers=-1)
        
        pbar.set_postfix({"error": "{:.3f}".format(np.sum(np.array(d)))})
        
        correspondences = np.vstack((np.arange(n), np.array(iq))).T
        correspondences = correspondences[d < max_distance, :]
        correspondences = o3d.utility.Vector2iVector(correspondences)
        
        if show_correspondences:
            line_set = o3d.geometry.LineSet().create_from_point_cloud_correspondences(object_local, scene, correspondences)
            o3d.visualization.draw_geometries([object_local, scene, line_set], window_name="ICP iteration")
        
        T = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(object_local, scene, correspondences)
        object_local = object_local.transform(T)
        if T_accum is None:
            T_accum = T
        else:
            T_accum = T_accum@T

    if show_result:
        o3d.visualization.draw_geometries([object_local, scene], window_name="result of ICP")

    return T_accum

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.listener_callback,
            1)
        
        # read target cad model from the package folder
        box_path = getAssetsPath() + '/target.stl'
        obj_mesh = o3d.io.read_triangle_mesh(box_path)
        
        # typical cad files are given in mm, so we need to convert to meters which is what the realsense uses
        obj_mesh.scale(1/1000, center=obj_mesh.get_center())
        self.object_pcd = obj_mesh.sample_points_uniformly(number_of_points=10000)
        #o3d.visualization.draw_geometries([self.object_pcd])

        # Demean the pointcloud to place it at the origin, i.e. center of the camera frame
        mean, cov = self.object_pcd.compute_mean_and_covariance()
        self.object_pcd = self.object_pcd.translate(-mean)
        
        self.object_pcd, self.object_features = preprocess_pcd(self.object_pcd)
        
        self.get_logger().info('Node has finished initialization')

    def broadcast_tf(self, T):
        t = TransformStamped()

        # header
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_depth_optical_frame'
        t.child_frame_id = 'pose_estimate'

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
        self.get_logger().info('I heard a pointcloud message')
        
        # read scene from ros message
        points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(points)

        scene_pcd, scene_features = preprocess_pcd(scene_pcd)
        
        # Fit plane to the scene to remove the table the object is sitting on
        plane_model, inliers = scene_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=10000)
        # plane model = ax + by + cz + d
        plane_normal = plane_model[0:3] # [a; b; c]
        plane_offset = plane_model[3] # d
        
        # remove everything from 0.01 meters above the plane (to account for noise) and below (to remove things that that are below the table)
        points = np.asarray(scene_pcd.points)
        distances = (points@plane_normal + plane_offset) / np.linalg.norm(plane_normal)
        inlier_points = points[distances < -0.01, :]
        inlier_points = o3d.utility.Vector3dVector(inlier_points)
        scene_pcd.points = inlier_points

        # do pose estimation using ransac and icp pointcloud registration
        kdtree = scp.spatial.KDTree(scene_pcd.points)
        T = global_alignment(scene_pcd, scene_features, self.object_pcd, self.object_features, kdtree, ransac_max_itr=5000, ransac_inlier_threshold=0.005)
        T = ICP(scene_pcd, self.object_pcd, kdtree, max_distance=0.01, max_itr=1000)@T
        
        # broadcast the pose to ros
        self.broadcast_tf(T)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()