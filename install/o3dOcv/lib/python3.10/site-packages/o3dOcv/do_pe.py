import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
import random
import sklearn as skln
import cProfile
import pstats

import scipy as scp

def do_pose_estimation(scene_pointcloud, object_pointcloud):
    #print("YOU NEED TO IMPLEMENT THIS!")

    scene_pointcloud = o3d.geometry.PointCloud(scene_pointcloud)
    object_pointcloud = o3d.geometry.PointCloud(object_pointcloud)
    scene_pointcloud.paint_uniform_color([0.8, 0.8, 0.8])
    object_pointcloud.paint_uniform_color([1, 0, 0])

    # Filter/Segment point cloud
    scene_pointcloud = spatial_filter(scene_pointcloud, p0=[-0.5, -0.1, -0.8], p1=[0.5, 0.3, -0.5], r=[0.45, 0, 0])

    # Use global pose estimation to find the object pose
    #profiler = cProfile.Profile()
    #profiler.enable()
    T = global_alignment(scene_pointcloud, object_pointcloud, ransac_max_itr=10000, feature_radius=0.03, normals_k_nearest=100, show_correspondences=False, show_result=False)
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('tottime')
    #stats.print_stats()
    aligned_object = object_pointcloud.transform(T)

    #print("GLOBAL ALLIGNMENT RESULT", T)

    # Use local pose estimation to refine the object pose
    #profiler = cProfile.Profile()
    #profiler.enable()
    T = ICP(scene_pointcloud, aligned_object)@T
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('tottime')
    #stats.print_stats()

    return T

def ICP(scene, object_local, max_itr = 1000, max_distance = 0.01, show_correspondences=False, show_result=False): #0.01
    object_local = o3d.geometry.PointCloud(object_local)
    scene = o3d.geometry.PointCloud(scene)
    
    kdtree = scp.spatial.KDTree(scene.points)
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

def global_alignment(scene, object, feature_radius = 0.02, ransac_max_itr = 5000, ransac_inlier_threshold = 0.005, normals_k_nearest = 100, show_best=False, show_correspondences=False, show_result=False): # low noise freature_radius=0.005 # normals_k_nearest=10 ransac_threshold=0.0005
    object = o3d.geometry.PointCloud(object)
    scene = o3d.geometry.PointCloud(scene)
    object.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(normals_k_nearest))
    scene.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(normals_k_nearest))

    # shape = (N, 33)
    object_features = np.asarray(
        o3d.pipelines.registration.compute_fpfh_feature(
            object,
            o3d.geometry.KDTreeSearchParamRadius(feature_radius)
        ).data.T
    )
    scene_features = np.asarray(
        o3d.pipelines.registration.compute_fpfh_feature(
            scene,
            o3d.geometry.KDTreeSearchParamRadius(feature_radius)
        ).data.T
    )
    
    # Match features between scene and object (euclidean distance in 33D feature space)
    correspondences = skln.metrics.pairwise_distances_argmin(object_features, scene_features) # f(X, Y) computes for each row in X the index of the row of Y which is closest
    correspondences = np.vstack((np.arange(object_features.shape[0]), correspondences)).T

    if show_correspondences:
        line_set = o3d.geometry.LineSet().create_from_point_cloud_correspondences(object, scene, o3d.utility.Vector2iVector(correspondences))
        o3d.visualization.draw_geometries([object, scene, line_set], window_name="correspondences")

    kdtree = scp.spatial.KDTree(scene.points)

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
        o3d.visualization.draw_geometries([object.transform(bestT), scene], window_name = 'Result of global alignment')

    return bestT

def spatial_filter(input_cloud, p0, p1, r, show_bounding_box=False):
    bb = o3d.geometry.AxisAlignedBoundingBox(p0, p1)
    obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bb) 
    obb.rotate(o3d.geometry.get_rotation_matrix_from_xyz(r))
    cropped = input_cloud.crop(obb)
    if show_bounding_box:
        display_bounding_box(cropped, obb)
    return cropped

def display_bounding_box(cloud, bb):
    # visualize bounding box
    bb.color = [0.082, 0.878, 0.078]
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(bb)
    vis.add_geometry(cloud)
    vis.run()

def display_removal(preserved_points, removed_points):
    removed_points.paint_uniform_color([1, 0, 0])        # Show removed points in red
    preserved_points.paint_uniform_color([0.8, 0.8, 0.8])# Show preserved points in gray
    o3d.visualization.draw_geometries([removed_points, preserved_points])