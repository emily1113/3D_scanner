import cv2
import cv2.aruco as aruco
import numpy as np


def get_transformation_matrix(rvec, tvec):
    # 將旋轉向量轉換為旋轉矩陣
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 創建4x4的剛體變換矩陣
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = tvec.flatten()

    return transformation_matrix

# 設置Aruco字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

# 載入影像
image = cv2.imread("C:/Users/ASUS/Desktop/POINT/red/ArUco/rgb_image_00000.png")

# 偵測Aruco標記
corners, ids, rejectedImgPoints = aruco.detectMarkers(image, aruco_dict)

# 如果偵測到標記
if ids is not None:
    print(f"偵測到的Aruco標記ID: {ids.flatten()}")

    # 設置相機內參（請根據您的相機數據替換）
    camera_matrix = np.array([[2424.2794834773194, 0, 963.0166430088262],
                              [0, 2424.6822621489446, 621.0569418586338],
                              [0, 0, 1]])
    dist_coeffs = np.zeros((5,))  # 假設無畸變

    # 偵測標記的姿態
    marker_length = 0.151  # 單位: 米 (需與真實Aruco大小一致)
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    # 遍歷每個偵測到的標記
    for i in range(len(ids)):
        # 畫出Aruco標記框和座標軸
        aruco.drawDetectedMarkers(image, corners)
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length)

        # 計算剛體變換矩陣
        transformation_matrix = get_transformation_matrix(rvecs[i], tvecs[i])
        print(f"Aruco ID: {ids[i][0]}")
        print(f"剛體變換矩陣:\n{transformation_matrix}")

    # 顯示結果
    cv2.imshow("Aruco Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未偵測到Aruco標記")
