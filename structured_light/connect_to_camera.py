# 從 MechEye 套件匯入必要的模組
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *


# 定義類別，用於發現並連接相機
class ConnectToCamera(object):
    def __init__(self):
        # 初始化相機物件
        self.camera = Camera()

    def main(self):
        print("正在搜尋所有可用的相機...")
        # 發現所有可用的相機
        camera_infos = Camera.discover_cameras()

        if len(camera_infos) == 0:
            print("未找到相機。")
            return

        # 顯示所有可用相機的資訊
        for i in range(len(camera_infos)):
            print("相機索引 :", i)
            print_camera_info(camera_infos[i])

        print("請輸入要連接的相機索引：")
        input_index = 0

        # 讓使用者輸入要連接的相機索引，並檢查輸入是否有效
        while True:
            input_index = input()
            if input_index.isdigit() and 0 <= int(input_index) < len(camera_infos):
                input_index = int(input_index)
                break
            print("輸入無效！請輸入有效的相機索引：")

        # 根據使用者選擇的索引連接相機
        error_status = self.camera.connect(camera_infos[input_index])
        if not error_status.is_ok():
            # 顯示錯誤訊息並結束
            show_error(error_status)
            return

        print("成功連接到相機。")

        # 斷開相機連接
        self.camera.disconnect()
        print("已成功斷開相機連接。")


# 程式進入點
if __name__ == '__main__':
    # 建立類別實例並執行主函式
    a = ConnectToCamera()
    a.main()
