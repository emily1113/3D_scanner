from plyfile import PlyData
import numpy as np
import matplotlib.pyplot as plt

# 讀取 PLY 檔案（plyfile 會自動判斷格式）
plydata = PlyData.read("C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00000.ply")
vertex_data = plydata['vertex'].data

# 將 x, y, z 座標堆疊成 NumPy 陣列
points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
print("下採樣前的點數：", points.shape[0])

# 均勻下採樣：每隔 5 個點選取一個
downsampled_points = points[::5]
print("下採樣後的點數：", downsampled_points.shape[0])

# 3D 可視化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(downsampled_points[:, 0], downsampled_points[:, 1], downsampled_points[:, 2], s=1)
ax.set_title("均勻下採樣後的點雲")
plt.show()
