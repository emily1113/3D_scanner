from mecheye.shared import *
from mecheye.area_scan_3d_camera import *

class ComputeNormalsFromPly(object):
    def __init__(self):
        self.frame_3d = Frame3D()

    # 讀取指定路徑的 PLY 點雲，計算法向量後儲存至新檔案
    def compute_normals_from_loaded_point_cloud(self, input_file, output_file):
        show_error(self.frame_3d.save_untextured_point_cloud_with_normals(FileFormat_PLY, output_file))



    def main(self):
        input_file = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply"      # 請修改為實際的輸入檔案路徑
        output_file = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001_out.ply"      # 請修改為實際的輸出檔案路徑
        self.compute_normals_from_loaded_point_cloud(input_file, output_file)

if __name__ == '__main__':
    a = ComputeNormalsFromPly()
    a.main()
