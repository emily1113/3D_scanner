# 使用此範例，你可以計算法線並保存帶有法線的點雲。法線可以在相機或電腦上計算。

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d


class CapturePointCloudWithNormals(object):
    def __init__(self):
        self.camera = Camera()
        self.frame_3d = Frame3D()

    # 在相機上計算點的法線，並將帶有法線的點雲保存到檔案中
    def capture_point_cloud_with_normals_calculated_on_camera(self):
        point_cloud_file = "PointCloud_1.ply"
        # 使用相機捕捉帶有法線的 3D 資料
        if self.camera.capture_3d_with_normal(self.frame_3d).is_ok():
            # 將點雲保存為帶有法線的 PLY 檔案
            show_error(
                self.frame_3d.save_untextured_point_cloud_with_normals(FileFormat_PLY, point_cloud_file))
            return True
        else:
            print("Failed to capture the point cloud.")  # 無法捕捉點雲
            self.camera.disconnect()
            print("Disconnected from the camera successfully.")  # 已成功斷開與相機的連接
            return False

    # 在電腦上計算點的法線，並將帶有法線的點雲保存到檔案中
    def capture_point_cloud_with_normals_calculated_locally(self):
        point_cloud_file = "PointCloud_2.ply"
        # 使用相機捕捉 3D 資料（無法線）
        if self.camera.capture_3d(self.frame_3d).is_ok():
            # 將點雲保存為帶有法線的 PLY 檔案
            show_error(
                self.frame_3d.save_untextured_point_cloud_with_normals(FileFormat_PLY, point_cloud_file))
            return True
        else:
            print("Failed to capture the point cloud.")  # 無法捕捉點雲
            self.camera.disconnect()
            print("Disconnected from the camera successfully.")  # 已成功斷開與相機的連接
            return False

    def main(self):
        # 嘗試連接到相機
        if find_and_connect(self.camera):
            # 確認是否可以進行 3D 捕捉
            if not confirm_capture_3d():
                return
            # 捕捉並保存帶有法線的點雲（由相機計算法線）
            if not self.capture_point_cloud_with_normals_calculated_on_camera():
                return
            # 捕捉並保存帶有法線的點雲（由電腦計算法線）
            if not self.capture_point_cloud_with_normals_calculated_locally():
                return
            # 斷開與相機的連接
            self.camera.disconnect()
            print("Disconnected from the camera successfully.")  # 已成功斷開與相機的連接


if __name__ == '__main__':
    a = CapturePointCloudWithNormals()
    a.main()
