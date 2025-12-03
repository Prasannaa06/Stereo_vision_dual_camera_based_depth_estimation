# Depth Estimation From Stereo Images

## Introduction:
This project implements a stereo-vision-based depth estimation system using two cameras placed at a fixed baseline distance. The system captures synchronized left–right images, computes disparity, and generates a depth map that can be used for robotics, autonomous driving, obstacle detection, and 3D scene understanding.

## Features:
- Uses two calibrated cameras for stereo image capture
- Performs camera calibration and image rectification
- Computes disparity maps using StereoBM / StereoSGBM
- Generates accurate depth estimation through triangulation
- Visualizes disparity and depth maps
- Easy-to-run Python scripts

## 🧠 How It Works

Stereo vision relies on identifying corresponding points in the left and right camera images.  
The shift between these points is known as **disparity**.

Depth is computed using:

$$
Depth = \frac{f \times B}{d}
$$

Where:  
- \( f \) = focal length  
- \( B \) = baseline distance between the two cameras  
- \( d \) = disparity

Incase of Stereo Setup, Depth estimation is dependent on disparity map.
![disparity drawio](https://user-images.githubusercontent.com/22910010/221393481-38847a4e-3c24-4daf-a803-e948051be575.png)

**Processing Pipeline:**
1. **Capture synchronized images** from two cameras  
2. **Calibrate cameras** to obtain intrinsic & extrinsic parameters  
3. **Undistort & rectify** images  
4. **Compute disparity map**  
5. **Convert disparity → depth map**  
6. **Visualize & analyze depth**  

## Technologies and Libraries Used:
- Python
- OpenCV
- NumPy
- Matplotlib
- YOLOv8 (For object Detection)
- Tensorflow

## Dependency

- Download Pre-Trained model which i shared at [Download Link](https://drive.google.com/file/d/1dBqxyEQm2g0bfUgWlTBNMo4opFtffedz/view?usp=sharing)

    Place it inside root folder and update the path in the config.py.

    ```
    RAFT_STEREO_MODEL_PATH = "pretrained_models/raft_stereo/raft-stereo_20000.pth"
    FASTACV_MODEL_PATH = "pretrained_models/fast_acvnet/kitti_2015.ckpt"
    ...
    ```

- Download YOLOv8 for object detection: from ultralytics import YOLO 

## Setting up DataSet
Download Kitti Dataset from [Download Link](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

 - Download Left/Right Images: Download stereo 2015/flow 2015/scene flow 2015 data set (2 GB)
 - Download Calibration files: Download calibration files (1 MB)

Keep these files in some path and update config.py


```
[config.py]
KITTI_CALIB_FILES_PATH=".../kitti_stereo_2015/data_scene_flow_calib/testing/calib_cam_to_cam/*.txt"
KITTI_LEFT_IMAGES_PATH=".../kitti_stereo_2015/testing/image_2/*.png"
KITTI_RIGHT_IMAGES_PATH=".../kitti_stereo_2015/testing/image_3/*.png"
...
```

## How to use


Run "python3 demo.py" change the configuration in config.py in order to run different architecture such as BGNet, CreStereo, RAFT-Stereo etc.

```
KITTI_CALIB_FILES_PATH=".../kitti_stereo_2015/data_scene_flow_calib/testing/calib_cam_to_cam/*.txt"
KITTI_LEFT_IMAGES_PATH=".../kitti_stereo_2015/testing/image_2/*.png"
KITTI_RIGHT_IMAGES_PATH=".../kitti_stereo_2015/testing/image_3/*.png"

RAFT_STEREO_MODEL_PATH = "pretrained_models/raft_stereo/raft-stereo_20000.pth"
FASTACV_MODEL_PATH = "pretrained_models/fast_acvnet/kitti_2015.ckpt"
DEVICE = "cuda"

# raft-stereo=0, fastacv-plus=1, bgnet=2, gwcnet=3, pasmnet=4, crestereo=5, hitnet=6, psmnet=7
ARCHITECTURE_LIST = ["raft-stereo", "fastacv-plus", "bgnet", 'gwcnet', 'pasmnet', 'crestereo', 'hitnet', 'psmnet']
ARCHITECTURE = ARCHITECTURE_LIST[1]
SAVE_POINT_CLOUD = 0
SHOW_DISPARITY_OUTPUT = 1
SHOW_3D_PROJECTION = 0
```

## Evaluation
Different state of the art (SOTA) deep learning based architetures are proposed to solve disparity and are given below:

![disparity_timeline drawio(1)](https://user-images.githubusercontent.com/22910010/221393628-17f66ca6-7255-45a4-8faf-46d768075a32.png)

Here is the profiling data:

![disparity_map_profile_](https://user-images.githubusercontent.com/22910010/221400837-5ad3ae24-f23f-420a-9b4d-8328c1499c21.png)

Here is the inference time on Nvidia-2080Ti

![inference drawio](https://user-images.githubusercontent.com/22910010/221400886-c5ed6e1b-1e7e-4bcd-b6d9-5709f4503863.png)

- [ ] Issue with HitNet Implementation.

# Acknowledgements
  Thanks to the authors of fastacv-plus, bgnet, gwcnet, pasmnet, crestereo, hitnet, psmnet and raft-stereo for their opensource code.
 
# References
- https://github.com/princeton-vl/RAFT-Stereo.git.
- https://github.com/gangweiX/Fast-ACVNet.
- https://github.com/3DCVdeveloper/BGNet.
- https://github.com/megvii-research/CREStereo.
- https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation.
- https://github.com/xy-guo/GwcNet.
- https://github.com/JiaRenChang/PSMNet.
- https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM/tree/master/PASMnet.

## Sample Output 
https://user-images.githubusercontent.com/22910010/221394203-adbb3581-5e6c-4edc-bfee-6f2469990896.mp4

(Note: Upper part is Disparity Map and bottom part is Object detection + Depth Estimation(z=?))

[PointCloud Output]

https://user-images.githubusercontent.com/22910010/220879862-f2a86b14-b30f-4f8a-9f2e-7fa07fc96d15.mp4

Full Video output is shared in YouTube [Link](https://youtu.be/cIse_5QOXx0?si=bjypo7xS2SbHLIsF)

---
Reach me @

[LinkedIn](https://www.linkedin.com/in/satya1507/) [GitHub](https://github.com/satya15july) [Medium](https://medium.com/@satya15july_11937)






