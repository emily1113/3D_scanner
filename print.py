import pandas as pd
from plyfile import PlyData, PlyElement
import numpy as np

# 指定 PLY 檔案路徑
ply_path = "C:/Users/ASUS/Desktop/ICP/ICP/red/1_40/point_cloud_00003.ply"
plydata = PlyData.read(ply_path)

# 取得頂點資料並轉換為適合 DataFrame 的格式
vertices = plydata['vertex']
data = [{'x': v['x'], 'y': v['y'], 'z': v['z']} for v in vertices]
df = pd.DataFrame(data)

# 去除含 NaN 的行
df = df.dropna()

# 計算每個點到原點的距離
df['distance_to_origin'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

# 找到距離最小的點
closest_to_origin = df.loc[df['distance_to_origin'].idxmin()]
origin_x, origin_y, origin_z = closest_to_origin['x'], closest_to_origin['y'], closest_to_origin['z']

print("最接近原點的點座標為：")
print(f"x: {origin_x}, y: {origin_y}, z: {origin_z}")

# 將最接近原點的點設為新原點，透過平移所有頂點
df['x'] = df['x'] - origin_x
df['y'] = df['y'] - origin_y
df['z'] = df['z'] - origin_z

# 將轉換後的點雲數據寫回新的 PLY 檔案
new_vertices = np.array([(row['x'], row['y'], row['z']) for _, row in df.iterrows()],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

new_ply_data = PlyData([PlyElement.describe(new_vertices, 'vertex')], text=True)
new_ply_path = "C:/Users/ASUS/Desktop/ICP/ICP/red/03.ply"
new_ply_data.write(new_ply_path)

print("轉換完成，新點雲文件已儲存至：", new_ply_path)
