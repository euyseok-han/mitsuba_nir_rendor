import torch
import cv2
import numpy as np
import open3d as o3d
from MiDaS.midas.dpt_depth import DPTDepthModel
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet

# 1. Load image
img = cv2.imread("dataset/Test_RGB/fea7bf06_00.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Load MiDaS
model = torch.hub.load("isl-org/MiDaS", "DPT_Large")
transform = torch.hub.load("isl-org/MiDaS", "transforms").dpt_transform

# 3. Predict depth
input_tensor = transform(img_rgb).unsqueeze(0)
depth_map = model(input_tensor)[0, :, :].detach().cpu().numpy()

# 4. Reproject to 3D
fx = fy = 500  # 가정된 focal length
cx, cy = img.shape[1] / 2, img.shape[0] / 2
points = []
for v in range(img.shape[0]):
    for u in range(img.shape[1]):
        Z = depth_map[v, u]
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points.append([X, Y, Z])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))
o3d.visualization.draw_geometries([pcd])

