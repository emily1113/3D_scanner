import cv2
import numpy as np

def extract_qr_corners(image):
    """
    提取 QR Code 的四個角點座標
    :param image: 輸入的 QR Code 圖片
    :return: 四個角點的坐標 (Nx2) 或 None 如果未檢測到 QR Code
    """
    qr_detector = cv2.QRCodeDetector()
    _, points, _ = qr_detector.detectAndDecode(image)
    if points is not None:
        return points[0]  # 提取四個角點
    return None

def calculate_rigid_transform(points1, points2):
    """
    計算剛體變換 (旋轉和平移)
    :param points1: 第一個 QR Code 的 3D 點 (Nx3)
    :param points2: 第二個 QR Code 的 3D 點 (Nx3)
    :return: 旋轉矩陣 (R) 和 平移向量 (T)
    """
    points1 = np.array(points1, dtype=np.float64)
    points2 = np.array(points2, dtype=np.float64)

    # 嘗試調用 estimateAffine3D
    result = cv2.estimateAffine3D(points1, points2)
    
    # 判斷返回值數量
    if len(result) == 3:  # 部分 OpenCV 版本只返回 3 個值
        retval, R, T = result
        if retval:
            return R, T
        else:
            raise ValueError("剛體變換計算失敗，請檢查點對是否正確")
    elif len(result) == 4:  # 其他版本可能返回 4 個值
        retval, R, T, inliers = result
        if retval:
            return R, T
        else:
            raise ValueError("剛體變換計算失敗，請檢查點對是否正確")
    else:
        raise RuntimeError("未知返回值結構，請檢查 OpenCV 版本")

# 載入兩張 QR Code 圖片
image1_path = "C:/Users/ASUS/Desktop/POINT/red/1_40/rgb_image_00000.png"  # 替換為你的第一張圖片路徑
image2_path = "C:/Users/ASUS/Desktop/POINT/red/1_40/rgb_image_00001.png"  # 替換為你的第二張圖片路徑

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# 提取兩張圖片中的 QR Code 角點
corners1 = extract_qr_corners(image1)
corners2 = extract_qr_corners(image2)

if corners1 is not None and corners2 is not None:
    # 假設 QR Code 是位於 Z=0 的平面，將 2D 點拓展為 3D 點 (添加 Z=0)
    points1_3D = np.hstack((corners1, np.zeros((4, 1))))
    points2_3D = np.hstack((corners2, np.zeros((4, 1))))

    # 計算剛體變換
    R, T = calculate_rigid_transform(points1_3D, points2_3D)

    # 輸出結果
    print("旋轉矩陣 (R):\n", R)
    print("平移向量 (T):\n", T)
else:
    print("未能檢測到 QR Code 的角點")
