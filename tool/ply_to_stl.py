import open3d as o3d

# 載入 PLY 檔案
input_ply = "C:/Users/ASUS/Desktop/POINT/3D_scanner/sac_ia_aligned.ply"  # 替換為 PLY 檔案路徑
output_stl = "C:/Users/ASUS/Desktop/POINT/3D_scanner/sac_bunny.stl"  # 替換為輸出的 STL 檔案路徑

try:
    mesh = o3d.io.read_triangle_mesh(input_ply)
    if mesh.is_empty():
        print("檔案讀取成功，但內容為空！請確認檔案內容。")
    else:
        o3d.io.write_triangle_mesh(output_stl, mesh)
        print(f"檔案已成功從 {input_ply} 轉換為 {output_stl}")
except Exception as e:
    print(f"載入檔案時發生錯誤：{e}")
