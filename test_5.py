from pclpy import pcl

# 读取pcd格式点云数据
pc = pcl.PointCloud.PointXYZ()
reader = pcl.io.PCDReader()
reader.read("bunny.pcd", pc)
# 可视化点云
viewer = pcl.visualization.CloudViewer("viewer")
viewer.showCloud(pc, "hi")
while not viewer.wasStopped(10):
    pass
