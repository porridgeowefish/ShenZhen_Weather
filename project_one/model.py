import torch.nn as nn
import torch
def define_model(X_train):
# --- 步骤 3: 定义模型、损失函数和优化器 ---
    try:
        print("\n开始定义模型、损失函数和优化器...")
        
        # 1. 定义线性回归模型
        # 首先，获取输入特征的数量
        num_features = X_train.shape[1]
        
        class LinearRegressionModel(nn.Module):
            def __init__(self, input_dim):
                super(LinearRegressionModel, self).__init__()
                # 定义一个线性层
                # 它会自动创建权重 W (shape: [input_dim, 1]) 和偏置 b (shape: [1])
                self.linear = nn.Linear(input_dim, 1)
                
            def forward(self, x):
                # 定义前向传播的路径：输入x通过线性层得到输出
                return self.linear(x)

        # 2. 实例化模型
        model = LinearRegressionModel(num_features)
        print("模型定义与实例化成功！")
        print(model) # 打印出模型的结构

        # 3. 定义损失函数
        # 使用均方误差损失 (Mean Squared Error)
        loss_fn = nn.MSELoss()
        print(f"\n损失函数: {loss_fn}")
        # 4. 定义优化器
        learning_rate = 0.0000000005  # 这是一个需要调整的超参数，我们先设一个较小的值
        wd = 0.0000001             # L2正则化(权重衰减)的系数lambda，也是超参数 
        # 使用随机梯度下降(SGD)优化器
        # model.parameters() 会自动将模型所有的可学习参数交给优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd)
        return model,loss_fn,optimizer
    except Exception as e:
        print(f"在定义模型、损失或优化器时发生错误: {e}")
