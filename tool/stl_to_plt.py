import trimesh
import os

# 使用者可在此直接指定 STL 與點雲（PLY）輸出路徑，以及採樣點數
INPUT_PATH = r"C:\Users\ASUS\Desktop\POINT\red\furiren\FRIEREN.stl"   # 將此處替換為你的 STL 檔案路徑
OUTPUT_PATH = r"C:\Users\ASUS\Desktop\POINT\red\furiren\FRIEREN_stl.ply"  # 將此處替換為你想輸出的 PLY 點雲路徑
SAMPLE_COUNT = 50000                        # 要從網格表面採樣的點數


def mesh_to_pointcloud(input_path: str, output_path: str, sample_count: int) -> None:
    """
    讀取指定路徑的 STL Mesh，從表面均勻採樣產生點雲，並匯出為 PLY 格式。

    :param input_path: 欲轉換的 STL Mesh 完整路徑
    :param output_path: 輸出 PLY 點雲完整路徑
    :param sample_count: 從 Mesh 表面採樣的點數
    """
    # 讀取 STL Mesh
    mesh = trimesh.load(input_path, force='mesh')
    print(f"Loaded mesh: {input_path}\n  - Vertices: {len(mesh.vertices)}\n  - Faces:    {len(mesh.faces)}")

    # 從表面均勻採樣
    samples, face_idx = trimesh.sample.sample_surface(mesh, sample_count)
    print(f"Sampled {len(samples)} points from mesh surface.")

    # 建立 PointCloud 物件並匯出 PLY
    cloud = trimesh.PointCloud(samples)
    cloud.export(output_path)
    print(f"已成功輸出點雲為: {output_path}")


def main():
    # 確保輸出資料夾存在
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    mesh_to_pointcloud(INPUT_PATH, OUTPUT_PATH, SAMPLE_COUNT)


if __name__ == "__main__":
    main()
