import torch
from matplotlib import pyplot as plt

# 加载保存的损失值和准确率
loss_v = torch.load("loss_v.pth")
acc_v = torch.load("acc_v.pth")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_v, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Iterations")
plt.legend()
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(acc_v, label="Training Accuracy", color="orange")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Iterations")
plt.legend()
plt.show()
