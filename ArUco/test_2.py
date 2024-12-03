import cv2
import numpy as np

# 加載相機內參數和失真係數（事先標定得到的）
camera_matrix = np.array([[2424.2794834773194, 0, 963.0166430088262],
                          [0, 2424.6822621489446, 621.0569418586338],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.array((0.0, 0.0, 0.0, 0.0, 0.0))

# 定義 ArUco 標誌的字典和標誌大小（以米為單位）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
marker_length = 0.05  # 標誌的邊長，假設為 5 厘米

# 初始化 ArUco 檢測參數
detector_params = cv2.aruco.DetectorParameters()

# 從固定圖像加載（這裡假設圖像位於 'image.jpg'）
image_path = "C:/Users/ASUS/Desktop/POINT/red/ArUco/rgb_image_00000.png"
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"無法讀取圖像，請檢查路徑是否正確: {image_path}")

# 檢測 ArUco 標誌
corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=detector_params)

# 如果檢測到標誌
if ids is not None:
    # 繪製檢測到的標誌
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # 計算每個檢測到的 ArUco 標誌的姿態
    for i in range(len(ids)):
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_length, camera_matrix, dist_coeffs)

        # 繪製標誌的姿態（坐標系）
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        # 輸出旋轉向量和平移向量
        print(f"ID: {ids[i][0]}")
        print(f"旋轉向量 (rvec): {rvec}")
        print(f"平移向量 (tvec): {tvec}")

        # 計算深度（相機坐標系中的 z 值）
        depth = tvec[0][0][2]
        print(f"ID: {ids[i][0]} 的深度: {depth} 米")

# 繪製相機坐標系（將原點作為參考）
origin = np.array([[0, 0, 0]], dtype=float)  # 相機坐標系的原點
rvec_origin = np.zeros((3, 1))  # 沒有旋轉
tvec_origin = np.zeros((3, 1))  # 相機原點
cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_origin, tvec_origin, 0.1)  # 繪製相機坐標系

# 標出相機坐標系
origin = (int(camera_matrix[0, 2]), int(camera_matrix[1, 2]))
cv2.circle(frame, origin, 5, (0, 0, 255), -1)
# 顯示影像
cv2.imshow('Aruco Detection', frame)

# 按下 'q' 鍵退出循環
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉視窗
cv2.destroyAllWindows()
