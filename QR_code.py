import numpy as np
import cv2

def calculate_rigid_transform(points1, points2):
    # 計算幾何中心
    center1 = np.mean(points1, axis=0)
    center2 = np.mean(points2, axis=0)

    # 中心化點集
    centered1 = points1 - center1
    centered2 = points2 - center2

    # 計算旋轉矩陣
    H = np.dot(centered1.T, centered2)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 確保旋轉矩陣的正確性
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 計算平移向量
    T = center2 - np.dot(R, center1)

    return R, T

# 讀取影像
image1 = cv2.imread("C:/Users/ASUS/Desktop/POINT/red/1_40/rgb_image_00000.png")
image2 = cv2.imread("C:/Users/ASUS/Desktop/POINT/red/1_40/rgb_image_00001.png")

# 初始化 QRCode 偵測器
qr_detector = cv2.QRCodeDetector()

# 偵測 QR Code 並獲取頂點座標
_, points1, _ = qr_detector.detectAndDecode(image1)
_, points2, _ = qr_detector.detectAndDecode(image2)

if points1 is not None and points2 is not None:
    # 提取頂點座標
    points1 = points1[0]  # QR Code 1 頂點
    points2 = points2[0]  # QR Code 2 頂點

    # 計算剛體變換
    R, T = calculate_rigid_transform(points1, points2)

    print("旋轉矩陣 (R):")
    print(R)
    print("\n平移向量 (T):")
    print(T)
else:
    print("未偵測到 QR Code")
