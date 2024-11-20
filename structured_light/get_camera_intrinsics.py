# 從 MechEye 套件匯入相關模組
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, print_camera_intrinsics


# 定義一個類別，用於獲取相機內部參數
class GetCameraIntrinsics(object):
    def __init__(self):
        # 初始化相機物件
        self.camera = Camera()
        # 初始化相機內部參數物件
        self.intrinsics = CameraIntrinsics()

    # 獲取並打印相機內部參數
    def get_device_intrinsic(self):
        # 從相機中獲取內部參數
        show_error(self.camera.get_camera_intrinsics(self.intrinsics))
        # 打印內部參數（焦距、光學中心等）
        print_camera_intrinsics(self.intrinsics)

    # 主函式：執行整個流程
    def main(self):
        # 嘗試連接到相機
        if find_and_connect(self.camera):
            # 獲取並打印相機內部參數
            self.get_device_intrinsic()
            # 斷開相機連接
            self.camera.disconnect()
            print("已成功斷開相機連接。")


# 程式進入點
if __name__ == '__main__':
    # 建立類別實例並執行主函式
    a = GetCameraIntrinsics()
    a.main()
