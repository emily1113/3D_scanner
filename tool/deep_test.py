import numpy as np
from tifffile import imread

depth = imread(r"C:\Users\ASUS\Desktop\POINT\red\furiren\depth_image_00001.tiff")
# print("所有點對相機的距離（原始值）:")
# print(depth.flatten())

# print("距離統計:")
# print("min:", np.min(depth))
# print("max:", np.max(depth))
# print("mean:", np.mean(depth))
print("深度圖 shape：", depth.shape)
print("最近距離 (min)：", np.min(depth))
print("最遠距離 (max)：", np.max(depth))
print("平均距離 (mean)：", np.mean(depth))
print("中位數 (median)：", np.median(depth))
print("標準差 (std)：", np.std(depth))