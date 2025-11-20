import cv2
import argparse
import glob
import numpy as np
import torch

from PIL import Image
import open3d as o3d
import config
import random
random.seed(42)
class_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(80)]


class Open3dVisualizer():
    def __init__(self, K):
        self.point_cloud = o3d.geometry.PointCloud()
        self.o3d_started = False
        self.K = K

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
    
    def __call__(self, rgb_image, depth_map, max_dist=20):
        self.update(rgb_image, depth_map, max_dist)

    def update(self, rgb_image, depth_map, max_dist=20):
        # Prepare the rgb image
        rgb_image_resize = cv2.resize(rgb_image, (depth_map.shape[1],depth_map.shape[0]))
        rgb_image_resize = cv2.cvtColor(rgb_image_resize, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_image_resize), 
                                                                   o3d.geometry.Image(depth_map),
                                                                   1, depth_trunc=max_dist*1000, 
                                                                   convert_rgb_to_intensity = False)
        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.K)
        temp_pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        # Add values to vectors
        self.point_cloud.points = temp_pcd.points
        self.point_cloud.colors = temp_pcd.colors

        # Add geometries if it is the first time
        if not self.o3d_started:
            self.vis.add_geometry(self.point_cloud)
            self.o3d_started = True

            # Set camera view
            ctr = self.vis.get_view_control()
            ctr.set_front(np.array([ -0.0053112027751292369, 0.28799919460714768, 0.95761592250270977 ]))
            ctr.set_lookat(np.array([-78.783105080589237, -1856.8182240774879, -10539.634663481682]))
            ctr.set_up(np.array([-0.029561736688513099, 0.95716567219818627, -0.28802774118017438]))
            ctr.set_zoom(0.31999999999999978)

        else:
            self.vis.update_geometry(self.point_cloud)

        self.vis.poll_events()
        self.vis.update_renderer()

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(config.DEVICE)

def get_calibration_parameters(file):
    parameters = []
    with open(file, 'r') as f:
        fin = f.readlines()
        for line in fin:
            #print("Line: {}".format(line))
            if line[:4] == 'K_02':
                parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
            elif line[:4] == 'T_02':
                parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
            elif line[:9] == 'P_rect_02':
                parameters.append(np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)) 
            elif line[:4] == 'K_03':
                parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1)) 
            elif line[:4] == 'T_03':
                parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
            elif line[:9] == 'P_rect_03':
                parameters.append(np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)) 
    return parameters

'''def find_distances(depth_map, pred_bboxes, img=None, method="median"):
    h, w = depth_map.shape[:2]
    depth_list = []
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box)
        print("Box:", x1, y1, x2, y2, "Size:", x2 - x1, y2 - y1)
        x1, x2 = np.clip([x1, x2], 0, w-1)
        y1, y2 = np.clip([y1, y2], 0, h-1)
        region = depth_map[y1:y2, x1:x2]
        # Use only valid depth (no nans/no zeros)
        valid = region[(np.isfinite(region)) & (region > 0.1)]
        if valid.size == 0:
            depth = 0.0
        elif method == "closest":
            depth = float(np.min(valid))
        elif method == "average":
            depth = float(np.mean(valid))
        else:  # default: median
            depth = float(np.median(valid))
        depth_list.append(depth)
    return depth_list'''
def find_distances(depth_map, pred_bboxes, img, method="center"):
    depth_list = []
    h, w, _ = img.shape
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box)
        # clamp coordinates
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        
        obstacle_depth = depth_map[y1:y2, x1:x2]
        if obstacle_depth.size == 0:
            depth_list.append(0)
            continue

        if method == "closest":
            depth_list.append(float(np.min(obstacle_depth)))
        elif method == "average":
            depth_list.append(float(np.mean(obstacle_depth)))
        elif method == "median":
            depth_list.append(float(np.median(obstacle_depth)))
        else:
            # center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_list.append(float(depth_map[cy, cx]))
    return depth_list




def calc_depth_map(disp_left, k_left, t_left, t_right):
    import torch
    if isinstance(disp_left, torch.Tensor):
        disp_left = disp_left.detach().cpu().numpy()
    # Get the focal length from the K matrix
    f = k_left[0, 0]
    # Get the distance between the cameras from the t matrices (baseline)
    b = abs(t_left[0] - t_right[0]) #On the setup page, you can see 0.54 as the distance between the two color cameras (http://www.cvlibs.net/datasets/kitti/setup.php)
    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1
    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disp_left.shape, np.single)
    # Calculate the depths 
    depth_map[:] = f * b / disp_left[:]
    print(f"Focal length f: {f}, Baseline (B): {b}")
    return depth_map

def draw_depth(depth_map, img_width, img_height, max_dist=10):
		
	return util_draw_depth(depth_map, (img_width, img_height), max_dist)

def draw_disparity(disparity_map, img_width, img_height):

    disparity_map =  cv2.resize(disparity_map,  (img_width, img_height))
    norm_disparity_map = 255*((disparity_map-np.min(disparity_map))/
                              (np.max(disparity_map)-np.min(disparity_map)))

    #return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, alpha=0.01),cv2.COLORMAP_JET)
    return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)

'''def add_depth(depth_list, result, pred_bboxes):
    h, w, _ = result.shape
    res = result.copy()
    for i, distance in enumerate(depth_list):
        cv2.line(res,(int(pred_bboxes[i][0]*w - pred_bboxes[i][2]*w/2),int(pred_bboxes[i][1]*h - pred_bboxes[i][3]*h*0.5)),(int(pred_bboxes[i][0]*w + pred_bboxes[i][2]*w*0.5),int(pred_bboxes[i][1]*h + pred_bboxes[i][3]*h*0.5)),(255,255,255),2)
        cv2.line(res,(int(pred_bboxes[i][0]*w + pred_bboxes[i][2]*w/2),int(pred_bboxes[i][1]*h - pred_bboxes[i][3]*h*0.5)),(int(pred_bboxes[i][0]*w - pred_bboxes[i][2]*w*0.5),int(pred_bboxes[i][1]*h + pred_bboxes[i][3]*h*0.5)),(255,255,255),2)
        cv2.putText(res, 'z={0:.2f} m'.format(distance), (int(pred_bboxes[i][0]*w - pred_bboxes[i][2]*w*0.5),int(pred_bboxes[i][1]*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)    
    return res'''

def add_depth(depth_list, result, pred_bboxes, labels, confidences, class_ids):
    h, w, _ = result.shape
    res = result.copy()
    if len(depth_list) == 0 or len(pred_bboxes) == 0:
        return res
    for i, distance in enumerate(depth_list):
        print(f"Box {i}, {labels[i]}: depth={distance}")
        x1, y1, x2, y2 = map(int, pred_bboxes[i])
        color = class_colors[int(class_ids[i])]
        text1 = f'{labels[i]}: {int(confidences[i]*100)}%'
        font_scale = 0.5
        thickness = 1
        # Draw only the box (now colored)
        cv2.rectangle(res, (x1, y1), (x2, y2), color, 2)
        # Label and confidence above box (now colored)
        cv2.putText(res, text1, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
        # Depth inside box, bottom left (kept red)
        text2 = f'z={distance:.2f} m'
        cv2.putText(res, text2, (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness, cv2.LINE_AA)
    return res



def util_draw_depth(depth_map, img_shape, max_dist):

	norm_depth_map = 255*(1-depth_map/max_dist)
	norm_depth_map[norm_depth_map < 0] = 0
	norm_depth_map[norm_depth_map >= 255] = 0
	norm_depth_map =  cv2.resize(norm_depth_map, img_shape)
	return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)

def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'ab') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
