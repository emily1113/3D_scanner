import cv2
import cv2.aruco as aruco
import numpy as np

# 摄像头内参矩阵和畸变系数（可以根据实际情况调整）
camera_matrix = np.array([[2424.2794834773194, 0, 963.0166430088262],
                          [0, 2424.6822621489446, 621.0569418586338],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.array((0.0, 0.0, 0.0, 0.0, 0.0))

def resize_image(image, max_width, max_height):
    """
    调整图像大小以适应屏幕范围。

    参数：
    - image: 原始图像
    - max_width: 最大宽度
    - max_height: 最大高度

    返回：
    - 调整后的图像
    """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale < 1:  # 只有当图像超过限制时才进行缩放
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image

def detect_aruco_from_image(image_path, dictionary=aruco.DICT_6X6_50, max_width=800, max_height=600):
    """
    从图片中检测 ArUco 标记，并在标记上绘制三维坐标系。

    参数：
    - image_path: 图片路径
    - dictionary: ArUco 字典
    - max_width: 显示图像的最大宽度
    - max_height: 显示图像的最大高度
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图片")
        return

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取 ArUco 字典和检测参数
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    parameters = aruco.DetectorParameters()

    # 检测 ArUco 标记
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # 如果检测到 ArUco 标记，估计其姿态并绘制坐标轴
    if ids is not None:
        image = aruco.drawDetectedMarkers(image, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            # 使用 cv2.drawFrameAxes 绘制坐标轴
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # 调整图像大小以适应屏幕
    image = resize_image(image, max_width, max_height)

    # 显示结果
    cv2.imshow('ArUco Markers with 3D Axis', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 替换图片路径
image_path = "C:/Users/ASUS/Desktop/POINT/red/ArUco/rgb_image_00000.png"
detect_aruco_from_image(image_path, max_width=800, max_height=600)


#image_path = "C:/Users/ASUS/Desktop/POINT/3D_scanner/ArUco/rgb_image_00000.png"