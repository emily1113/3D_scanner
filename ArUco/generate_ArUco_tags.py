import cv2
import cv2.aruco as aruco
import numpy as np

def generate_colored_aruco_marker(marker_id, marker_size=151, dictionary=aruco.DICT_6X6_50, marker_color=(0, 0, 255), trim_line_offset_cm=3, dpi=96, line_color=(0, 0, 0)):
    """
    生成自定義顏色的 ArUco 標記圖像，並添加修剪線

    :param marker_id: 標記 ID
    :param marker_size: 標記像素大小
    :param dictionary: ArUco 字典
    :param marker_color: 標記顏色 (B, G, R)
    :param trim_line_offset_cm: 修剪線距離標記的偏移量（單位：公分）
    :param dpi: 圖像 DPI (影響 px/cm 比例)
    :param line_color: 修剪線顏色（默認黑色）
    """
    trim_line_offset_px = int((dpi / 2.54) * trim_line_offset_cm)  # 1cm = dpi / 2.54 像素

    # 獲取指定的字典
    aruco_dict = aruco.getPredefinedDictionary(dictionary)

    # 生成標記圖像 (二值圖像，黑色部分為0，白色部分為255)
    img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # 將二值圖像轉換為彩色 (BGR)
    img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 將黑色部分替換為指定顏色
    img_colored[np.where((img_colored == [0, 0, 0]).all(axis=2))] = marker_color

    # 在圖像四周添加白邊
    border_width = trim_line_offset_px + 30  # 修剪線外加 30px 白邊
    img_with_border = cv2.copyMakeBorder(img_colored, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # 繪製修剪線
    height, width, _ = img_with_border.shape
    points = [
        (trim_line_offset_px, trim_line_offset_px),
        (width - trim_line_offset_px, trim_line_offset_px),
        (width - trim_line_offset_px, height - trim_line_offset_px),
        (trim_line_offset_px, height - trim_line_offset_px),
    ]
    cv2.polylines(img_with_border, [np.array(points)], isClosed=True, color=line_color, thickness=2)

    # 保存圖像
    output_file = f'aruco_marker_{marker_id}_colored_with_trim_line.png'
    cv2.imwrite(output_file, img_with_border)
    print(f"已保存自定義顏色的 ArUco 標記 ID {marker_id} 為 {output_file}")

# 生成 ID 為 0 的 ArUco 標記，將黑色替換為紅色，並添加修剪線
generate_colored_aruco_marker(marker_id=2, marker_color=(0, 0, 255), trim_line_offset_cm=1)
