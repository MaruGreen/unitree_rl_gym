import torch
import time

# 载入模型
model_path = '../pre_train/g1/motion.pt'  # 替换为你的模型文件路径
model = torch.jit.load(model_path)

# model_path = './mlp_model.pt'  # 替换为你的模型文件路径
# model = torch.load(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device {device}')
# device = 'cpu'
model.to(device)

# 准备输入数据
input_data = torch.randn(1, 47).to(device)  # 示例输入

model.eval()  # 设置模型为评估模式
for _ in range(100):
    # 进行推理并测量时间
    start_time = time.monotonic()  # 记录开始时间
    with torch.no_grad():  # 不需要计算梯度
        output = model(input_data)  # 进行推理
    end_time = time.monotonic()  # 记录结束时间

    # 计算推理时间
    inference_time = end_time - start_time
    # print(f'Inference output: {output}')
    print(f'Inference time: {inference_time:.6f} seconds')
