# save_model.py
import torch
from model import DCP

# 初始化模型
model = DCP()

# 保存模型
torch.save(model.state_dict(), "dcp_model.pth")
print("模型已保存至 dcp_model.pth")

# 載入模型
model.load_state_dict(torch.load("dcp_model.pth"))
model.eval()
print("模型已成功載入")
