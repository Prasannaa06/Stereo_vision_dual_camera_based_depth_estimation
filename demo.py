import os
import cv2
import time
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d

from utils import get_calibration_parameters, calc_depth_map, find_distances, add_depth, Open3dVisualizer, write_ply
from object_detector import ObjectDetectorAPI
from disparity_estimator.raftstereo_disparity_estimator import RAFTStereoEstimator
from disparity_estimator.fastacv_disparity_estimator import FastACVEstimator
from disparity_estimator.bgnet_disparity_estimator import BGNetEstimator
from disparity_estimator.gwcnet_disparity_estimator import GwcNetEstimator
from disparity_estimator.pasmnet_disparity_estimator import PASMNetEstimator
from disparity_estimator.crestereo_disparity_estimator import CREStereoEstimator
from disparity_estimator.psmnet_disparity_estimator import PSMNetEstimator
from disparity_estimator.hitnet_disparity_estimator import HitNetEstimator
import config
import random
random.seed(42)
class_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(80)]



#def visualize_point_cloud(ply_file):
    #pcd = o3d.io.read_point_cloud(ply_file)
    #o3d.visualization.draw_geometries([pcd])

def demo():
    if config.PROFILE_FLAG:
        disp_estimator = None
        if config.ARCHITECTURE == 'raft-stereo':
            disp_estimator = RAFTStereoEstimator()
        elif config.ARCHITECTURE == 'fastacv-plus':
            disp_estimator = FastACVEstimator()
        elif config.ARCHITECTURE == 'bgnet':
            disp_estimator = BGNetEstimator()
        elif config.ARCHITECTURE == 'gwcnet':
            disp_estimator = GwcNetEstimator()
        elif config.ARCHITECTURE == 'pasmnet':
            disp_estimator = PASMNetEstimator()
        elif config.ARCHITECTURE == 'crestereo':
            disp_estimator = CREStereoEstimator()
        elif config.ARCHITECTURE == 'psmnet':
            disp_estimator = PSMNetEstimator()
        elif config.ARCHITECTURE == 'hitnet':
            disp_estimator = HitNetEstimator()
        disp_estimator.profile()
        exit()

    left_images = sorted(glob.glob(config.KITTI_LEFT_IMAGES_PATH, recursive=True))
    right_images = sorted(glob.glob(config.KITTI_RIGHT_IMAGES_PATH, recursive=True))
    calib_files = sorted(glob.glob(config.KITTI_CALIB_FILES_PATH, recursive=True))
    index = 0
    init_open3d = False
    disp_estimator = None
    print("Disparity Architecture Used: {} ".format(config.ARCHITECTURE))
    if config.ARCHITECTURE == 'raft-stereo':
        disp_estimator = RAFTStereoEstimator()
    elif config.ARCHITECTURE == 'fastacv-plus':
        disp_estimator = FastACVEstimator()
    elif config.ARCHITECTURE == 'bgnet':
         disp_estimator = BGNetEstimator()
    elif config.ARCHITECTURE == 'gwcnet':
        disp_estimator = GwcNetEstimator()
    elif config.ARCHITECTURE == 'pasmnet':
        disp_estimator = PASMNetEstimator()
    elif config.ARCHITECTURE == 'crestereo':
        disp_estimator = CREStereoEstimator()
    elif config.ARCHITECTURE == 'psmnet':
        disp_estimator = PSMNetEstimator()
    elif config.ARCHITECTURE == 'hitnet':
        disp_estimator = HitNetEstimator()

    if config.SHOW_DISPARITY_OUTPUT:
        window_name = "Estimated depth with {}".format(config.ARCHITECTURE)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    sample_img = cv2.imread(left_images[0])
    h, w = sample_img.shape[:2]
    video_height, video_width = h * 2, w  # images are stacked vertically
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (video_width, video_height))
    
    for (imfile1, imfile2, calib_file) in tqdm(list(zip(left_images, right_images, calib_files))):
        img = cv2.imread(imfile1)
        parameters = get_calibration_parameters(calib_file)

        obj_det = ObjectDetectorAPI()
        start = time.time()
        result, pred_bboxes, class_ids, confidences = obj_det.predict(img)
        end = time.time()
        elapsed_time = (end - start) * 1000
        print("Evaluation Time for Object Detection with YOLO is : {} ms ".format(elapsed_time))

        COCO_CLASSES = [
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",
            "stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
            "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
            "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass",
            "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
            "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
            "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
            "scissors","teddy bear","hair drier","toothbrush"
        ]
        labels = [COCO_CLASSES[int(c)] for c in class_ids]


        print(pred_bboxes)

        start_d = time.time()
        disparity_map = disp_estimator.estimate(imfile1, imfile2)
        end_d = time.time()
        elapsed_time_d = (end_d - start_d) * 1000
        print("Evaluation Time for Disparity Estimation with {} is : {} ms ".format(config.ARCHITECTURE, elapsed_time_d))

        print("disparity_map:", disparity_map)
        disparity_left = disparity_map

        k_left = parameters[0]
        t_left = parameters[1]
        p_left = parameters[2]

        k_right = parameters[3]
        t_right = parameters[4]
        p_right = parameters[5]
        print("k_left:{}, t_left:{}".format(k_left, t_left))
        print("k_right:{}, t_right:{}".format(k_right, t_right))
        print("p_left:{}, p_right:{}".format(p_left, p_right))

        
        for i, disp in enumerate(disparity_map):
        # 1. Calculate and Squeeze Depth Map
            depth_map = calc_depth_map(disp, k_left, t_left, t_right)
            depth_map_2d = np.squeeze(depth_map)
            h_img, w_img = img.shape[:2]
            if depth_map_2d.shape != (h_img, w_img):
                print(f"Resizing depth_map_2d from {depth_map_2d.shape} to {(h_img, w_img)}")
                depth_map_2d = cv2.resize(depth_map_2d, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            print("depth_map_2d.shape after resize:", depth_map_2d.shape)  # This should print (h_img, w_img)

            print("depth_map_2d shape before find_distances:", depth_map_2d.shape)

            print(f"Depth map {i} shape: {depth_map_2d.shape}")  # Debug: Should be (H, W)

            # 2. Use Only 2D Depth Map for Processing
            depth_list = find_distances(depth_map_2d, pred_bboxes, img, method="center")
            if depth_list is None:
                print("ERROR: find_distances returned None, should be a list of depths!")
                depth_list = []
            res = add_depth(depth_list, result, pred_bboxes, labels, confidences, class_ids)


            # 3. Proceed with visualization, saving, point cloud, etc. using depth_map_2d and depth_list
            # Example:
            color_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_map_2d, alpha=0.01), cv2.COLORMAP_JET)

            # If you want interactive window exit (optional)
            if cv2.waitKey(1) == ord('q'):
                break



        if isinstance(disparity_map, list):
            disparity_map_scaled = [
                (dm.detach().cpu().numpy() * 256.).astype(np.uint16)
                if isinstance(dm, torch.Tensor)
                else (np.array(dm) * 256.).astype(np.uint16)
                for dm in disparity_map
            ]
        else:
            dm = disparity_map
            if isinstance(dm, torch.Tensor):
                dm = dm.detach().cpu().numpy()
            disparity_map_scaled = (dm * 256.).astype(np.uint16)

        # For ALL images (your 'all' request):
        color_depth_list = []
        for i, disp_img in enumerate(disparity_map_scaled if isinstance(disparity_map_scaled, list) else [disparity_map_scaled]):
            disp_img_2d = np.squeeze(disp_img)
            if disp_img_2d.ndim > 2:  # pick first channel if needed
                disp_img_2d = disp_img_2d[0]
            color_img = cv2.applyColorMap(cv2.convertScaleAbs(disp_img_2d, alpha=0.01), cv2.COLORMAP_JET)
            color_depth_list.append(color_img)

            # Optionally display or save:
            #cv2.imshow(f"Color Depth {i}", color_img)
            #cv2.imwrite(f"color_depth_{i}.png", color_img)

        depth_list = find_distances(depth_map_2d, pred_bboxes, img, method="center")

        res = add_depth(depth_list, result, pred_bboxes, labels, confidences, class_ids)
        print("img.shape {}".format(img.shape))
        #print("color_depth.shape {}".format(color_depth.shape))
        color_depth = color_depth_list[0]
        print("color_depth_list[0].shape {}".format(color_depth_list[0].shape))
        print("res.shape {}".format(res.shape))
        h = img.shape[0]
        w = img.shape[1]
        color_depth = cv2.resize(color_depth, (w, h))
        print("color_depth.shape after resize {}".format(color_depth.shape))
        combined_image = np.vstack((color_depth, res))
        print("combined_image shape:", combined_image.shape)
        out.write(combined_image)
        if config.SHOW_DISPARITY_OUTPUT:
            cv2.imshow(window_name, combined_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if config.SHOW_3D_PROJECTION:
            if init_open3d == False:
                w = img.shape[1]
                h = img.shape[0]
                print("w:{}, h: {}".format(w, h))
                print("kleft[0][0]: {}".format(k_left[0][0]))
                print("kleft[1][2]: {}".format(k_left[1][1]))
                print("kleft[1][2]: {}".format(k_left[0][2]))
                print("kleft[1][2]: {}".format(k_left[1][2]))
                print("kLeft: {}".format(k_left))

                K = o3d.camera.PinholeCameraIntrinsic(width=w,
                                                      height=h,
                                                      fx=k_left[0, 0],
                                                      fy=k_left[1, 1],
                                                      cx=k_left[0][2],
                                                      cy=k_left[1][2])
                open3dVisualizer = Open3dVisualizer(K)
                init_open3d = True
            open3dVisualizer(img, depth_map * 1000)

            o3d_screenshot_mat = open3dVisualizer.vis.capture_screen_float_buffer()
            o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
            o3d_screenshot_mat = cv2.cvtColor(o3d_screenshot_mat, cv2.COLOR_RGB2BGR)
        if config.SAVE_POINT_CLOUD:
            # Calculate depth-to-disparity
            cam1 = k_left  # left image - P2
            cam2 = k_right  # right image - P3

            print("p_left: {}".format(p_left))
            print("cam1:{}".format(cam1))

            Tmat = np.array([0.54, 0., 0.])
            Q = np.zeros((4, 4))
            cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2,
                              distCoeffs1=0, distCoeffs2=0,
                              imageSize=img.shape[:2],
                              R=np.identity(3), T=Tmat,
                              R1=None, R2=None,
                              P1=None, P2=None, Q=Q)

            print("Disparity To Depth")
            print(Q)
            print("disparity_left.shape: {}".format(disparity_left.shape))
            print("disparity_left: {}".format(disparity_left))

            points = cv2.reprojectImageTo3D(disparity_left.copy(), Q)
            # reflect on x axis

            reflect_matrix = np.identity(3)
            reflect_matrix[0] *= -1
            points = np.matmul(points, reflect_matrix)

            img_left = cv2.imread(imfile1)
            colors = cv2.cvtColor(img_left.copy(), cv2.COLOR_BGR2RGB)
            print("colors.shape: {}".format(colors.shape))
            disparity_left = cv2.resize(disparity_left, (colors.shape[1], colors.shape[0]))
            points = cv2.resize(points, (colors.shape[1], colors.shape[0]))
            print("points.shape: {}".format(points.shape))
            print("After mod. disparity_left.shape: {}".format(disparity_left.shape))
            # filter by min disparity
            mask = disparity_left > disparity_left.min()
            out_points = points[mask]
            out_colors = colors[mask]

            out_colors = out_colors.reshape(-1, 3)
            path_ply = os.path.join("output/point_clouds/", config.ARCHITECTURE)
            isExist = os.path.exists(path_ply)
            if not isExist:
                os.makedirs(path_ply)
            print("path_ply: {}".format(path_ply))

            file_name = path_ply + "/" +str(index) + ".ply"
            print("file_name: {}".format(file_name))
            write_ply(file_name, out_points, out_colors)
            index = index + 1
        
        if config.SHOW_DISPARITY_OUTPUT:
            if cv2.waitKey(1) == ord('q'):
                break
    out.release()
    
    if config.SHOW_DISPARITY_OUTPUT:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    demo()


