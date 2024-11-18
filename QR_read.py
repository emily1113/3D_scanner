import cv2

# 載入影像
image_path = "C:/Users/ASUS/Desktop/POINT/red/1118/rgb_image_00000.png"  # 替換為你的影像路徑
image = cv2.imread(image_path)

# 初始化 QRCode 偵測器
qr_detector = cv2.QRCodeDetector()

# 偵測並解碼 QR Code
data, points, _ = qr_detector.detectAndDecode(image)

if points is not None:
    # 將 QR Code 框線繪製在影像上
    points = points[0]  # 將多維陣列降一維

    for i in range(4):
        start_point = tuple(points[i])
        end_point = tuple(points[(i + 1) % 4])
        cv2.line(image, (int(start_point[0]), int(start_point[1])), 
                 (int(end_point[0]), int(end_point[1])), (0, 255, 0), 2)

        # 標示頂點座標
        cv2.putText(image, f"{i+1}: ({int(start_point[0])}, {int(start_point[1])})", 
                    (int(start_point[0]), int(start_point[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 計算中心點
    center_x = sum(point[0] for point in points) / 4
    center_y = sum(point[1] for point in points) / 4
    print(f"QR Code 資料: {data}")
    print(f"QR Code 頂點座標: {points}")
    print(f"QR Code 中心點: ({center_x}, {center_y})")

    # 繪製中心點
    cv2.circle(image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

    # 標示中心點座標
    cv2.putText(image, f"Center: ({int(center_x)}, {int(center_y)})", 
                (int(center_x), int(center_y) + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
else:
    print("未偵測到 QR Code")

# 顯示結果
cv2.imshow("QR Code Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
