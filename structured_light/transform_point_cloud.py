from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect, confirm_capture_3d
import numpy as np

# 定義一個點雲轉換的類別
class TransformPointCloud(object):
    def __init__(self):
        # 初始化相機物件
        self.camera = Camera()
        # 初始化 2D 和 3D 框架物件
        self.frame_all_2d_3d = Frame2DAnd3D()

    # 獲取並儲存轉換後的無紋理點雲
    def get_transformed_point_cloud(self):
        # 獲取點雲的座標系轉換參數
        transformation = get_transformation_params(self.camera)
        # 取得 3D 框架中的無紋理點雲
        frmae_3d = Frame3D(self.frame_all_2d_3d.frame_3d())
        transformed_point_cloud  =  transform_point_cloud(transformation, frmae_3d.get_untextured_point_cloud())
        # 設定輸出檔案名稱
        point_cloud_file = "PointCloud.ply"
        # 儲存點雲檔案
        show_error(Frame3D.save_point_cloud(transformed_point_cloud, FileFormat_PLY, point_cloud_file))
        print("捕獲並儲存點雲：{}.".format(point_cloud_file))

    # 獲取並儲存轉換後的有紋理點雲
    def get_transformed_textured_point_cloud(self):
        # 獲取座標系轉換參數
        transformation = get_transformation_params(self.camera)
        # 轉換帶紋理的點雲
        transformed_textured_point_cloud  = transform_textured_point_cloud(transformation, self.frame_all_2d_3d.get_textured_point_cloud())
        # 設定輸出檔案名稱
        textured_point_cloud_file = "TexturedPointCloud.ply"
        # 儲存點雲檔案
        show_error(Frame2DAnd3D.save_point_cloud(transformed_textured_point_cloud, FileFormat_PLY, textured_point_cloud_file))
        print("捕獲並儲存帶紋理的點雲：{}".format(textured_point_cloud_file))

    # 獲取並儲存轉換後的帶法向量點雲
    def get_transformed_point_cloud_with_normals(self):
        # 獲取座標系轉換參數
        transformation = get_transformation_params(self.camera)
        # 取得 3D 框架中的無紋理點雲並轉換法向量
        frmae_3d = Frame3D(self.frame_all_2d_3d.frame_3d())
        transformed_point_cloud_with_normals  =  transform_point_cloud_with_normals(transformation, frmae_3d.get_untextured_point_cloud())
        # 設定輸出檔案名稱
        point_cloud_with_normals_file = "PointCloudWithNormals.ply"
        # 儲存點雲檔案
        show_error(Frame3D.save_point_cloud_with_normals(transformed_point_cloud_with_normals, FileFormat_PLY, point_cloud_with_normals_file, False))
        print("捕獲並儲存帶法向量的點雲：{}.".format(point_cloud_with_normals_file))

    # 獲取並儲存轉換後的帶法向量和紋理的點雲
    def get_transformed_textured_point_cloud_with_normals(self):
        # 獲取座標系轉換參數
        transformation = get_transformation_params(self.camera)
        # 轉換帶法向量和紋理的點雲
        transformed_textured_point_cloud_with_normals  =  transform_textured_point_cloud_with_normals(transformation, self.frame_all_2d_3d.get_textured_point_cloud())
        # 設定輸出檔案名稱
        textured_point_cloud_with_normals_file = "TexturedPointCloudWithNormals.ply"
        # 儲存點雲檔案
        show_error(Frame2DAnd3D.save_point_cloud_with_normals(transformed_textured_point_cloud_with_normals, FileFormat_PLY, textured_point_cloud_with_normals_file, False))
        print("捕獲並儲存帶法向量和紋理的點雲：{}".format(textured_point_cloud_with_normals_file))

    # 主函式：處理點雲的整個流程
    def main(self):
        # 嘗試連接相機
        if find_and_connect(self.camera):
            # 確認相機是否可以捕捉 3D 資料
            if not confirm_capture_3d():
                return
            # 捕捉 2D 和 3D 資料
            show_error(self.camera.capture_2d_and_3d(self.frame_all_2d_3d))
            # 獲取相機到自定義座標系的剛體轉換參數
            transformation = get_transformation_params(self.camera)
            if transformation.__is__valid__() == False:
                print("轉換參數尚未設定，請使用客戶端工具設定自定義座標系統參數。")
                return
            # 執行點雲的轉換和儲存
            self.get_transformed_point_cloud()
            self.get_transformed_textured_point_cloud()
            self.get_transformed_point_cloud_with_normals()
            self.get_transformed_textured_point_cloud_with_normals()
            # 斷開相機連接
            self.camera.disconnect()
            print("已成功斷開相機連接。")

# 程式進入點
if __name__ == '__main__':
    a = TransformPointCloud()
    a.main()
