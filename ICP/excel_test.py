import pandas as pd

# 建立一個簡單的 DataFrame，只有一欄「City」和一筆測試城市
df_test = pd.DataFrame({
    "City": ["Taipei"]
})

# 將 DataFrame 輸出到 Excel 檔案
test_excel_path = r"C:\Users\user\Desktop\PointCloud\red\test_city.xlsx"
df_test.to_excel(test_excel_path, index=False)

# 顯示結果
df_test
