import cv2
import cv2.aruco as aruco

# 指定 ArUco 字典
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# 標記的 ID 和尺寸 (例如 ID=1, 標記大小為 200x200 像素)
marker_id = 1
marker_size = 200

# 生成 ArUco 標記
marker = aruco.drawMarker(aruco_dict, marker_id, marker_size)

# 保存標記為圖片
output_path = "aruco_marker_id_1.png"
cv2.imwrite(output_path, marker)

# 顯示生成的標記
cv2.imshow("ArUco Marker", marker)
cv2.waitKey(0)
cv2.destroyAllWindows()
