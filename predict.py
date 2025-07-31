import pandas as pd
import torch
from lunardate import LunarDate

file_path = 'D:\\python_file\\AI\\datasets\\DData.csv'

# a. 精确定义12列的名称
column_names = [
    'cumulative_rainfall', 'avg_temp', 'max_humidity', 'min_humidity',
    'avg_pressure', 'min_pressure', 'max_pressure', 'avg_humidity_alt',
    'min_temp', 'max_temp_raw', 'date'
]
# b. 加载数据，强制指定没有索引列
df = pd.read_csv(file_path, header=0, index_col=False, names=column_names,dtype={'date': str} )# 强制将名为'date'的列(我们新命名的)读作字符串) #指定行列开头
    
# c. 强制将所有“应该”是数值的列转换为数值类型
cols_to_numeric = [
    'cumulative_rainfall', 'avg_temp', 'max_humidity', 'min_humidity',
    'avg_pressure', 'min_pressure', 'max_pressure', 'avg_humidity_alt',
    'min_temp', 'max_temp_raw'
]
df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')


df['cumulative_rainfall'].fillna(0, inplace=True)
# d. 强制将日期列转换为字符串
df['date'] = df['date'].astype(str)

# --- 步骤 1.2: 数值修正与特征工程 ---
try:
    print("\n开始进行数值修正和特征工程...")
    
    # 1. 修正被放大了10倍的数值
    # 请根据你的新列名确认这个列表是否正确
    cols_to_divide = [
        'cumulative_rainfall','avg_temp', 'avg_pressure', 'min_pressure', 
        'max_pressure', 'min_temp', 'max_temp_raw'
    ]
    df[cols_to_divide] = df[cols_to_divide] / 10.0
    
    # 为了方便，我们可以把清理后的列重命名
    df.rename(columns={
        'max_temp_raw': 'max_temp',
    }, inplace=True)
    print("数值修正完成。")
    
    # 2. 创建新的气压特征
    # 特征一：气压差
    df['pressure_diff'] = df['max_pressure'] - df['min_pressure']
    
    # 特征二：气压偏离度 (Z-score 标准化)
    mean_pressure = df['avg_pressure'].mean()
    std_pressure = df['avg_pressure'].std()
    df['pressure_deviation'] = (df['avg_pressure'] - mean_pressure) / std_pressure
    
    print("新的气压特征创建成功！")
    
except Exception as e:
    print(f"在数值修正或特征工程时发生错误: {e}")


# --- 步骤 1.3: 日期转换、季节判断与独热编码 ---
try:
    print("\n开始进行日期处理和季节特征的独热编码...")
    
    date_as_int64 = df['date'].astype('int64')
    # 2. 通过精确的整数除法，得到YYYYMMDD
    date_yyyymmdd = date_as_int64 // 1000000
    # 3. 将YYYYMMDD整数转换为datetime对象
    df['date_dt'] = pd.to_datetime(date_yyyymmdd.astype(str), format='%Y%m%d')
    
    
#  get_season 函数 ---

    def get_season(date_obj):
        # 新增：从Timestamp中提取出纯日期部分
        date_only = date_obj.date()
        
        year = date_only.year
        try:
            # 计算节气时，也确保我们得到的是.date()对象
            lichun = LunarDate(year, 1, 1).toSolarDate()
            lixia = LunarDate(year, 4, 1).toSolarDate()
            liqiu = LunarDate(year, 7, 1).toSolarDate()
            lidong = LunarDate(year, 10, 1).toSolarDate()
            
            # 比较时，统一使用 date_only
            if date_only < lichun: return 'winter'
            elif lichun <= date_only < lixia: return 'spring'
            elif lixia <= date_only < liqiu: return 'summer'
            elif liqiu <= date_only < lidong: return 'autumn'
            else: return 'winter'
        except (ValueError, OverflowError): 
            return 'unknown'

    # 3. 创建'season'文本列
    df['season'] = df['date_dt'].apply(get_season)
    df = df[df['season'] != 'unknown'].copy()
    
    # 4. 对'season'列进行独热编码
    season_dummies = pd.get_dummies(df['season'], prefix='is')
    
    # 5. 合并独热编码列，并清理不再需要的中间列
    df = pd.concat([df, season_dummies], axis=1)
    df.drop(['date', 'date_dt', 'season'], axis=1, inplace=True)
    
    print("季节特征工程和独热编码完成！")
    print(df.head())
    print(df.describe())  # 输出数据的统计信息以验证处理结果


except Exception as e:
    print(f"在季节转换或独热编码时发生错误: {e}")



# 导入我们需要的工具
from sklearn.model_selection import train_test_split

# --- 步骤 2: 分割数据集 ---
try:
    print("\n开始分割数据集...")
    
    # 1. 确定特征(X)和目标(y)
    # 目标 y 是 'max_temp'
    y = df['max_temp']
    # 特征 X 是除了 'max_temp' 之外的所有列
    X = df.drop('max_temp', axis=1)
    
    # 2. 使用 train_test_split 进行分割
    # test_size=0.2 表示测试集占20%
    # random_state=42 保证每次分割结果都一样，方便复现
    X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("数据集分割完成！")
    print(f"训练集大小: {len(X_train_pd)} 行")
    print(f"测试集大小: {len(X_test_pd)} 行")
    X_train_pd = X_train_pd.astype('float32')
    X_test_pd = X_test_pd.astype('float32')
    # 3. 将Pandas数据转换为PyTorch Tensors
    # 使用 .values 可以获取底层的NumPy数组，再转换为Tensor
    # 指定dtype为float32，这是神经网络常用的类型
    X_train = torch.tensor(X_train_pd.values, dtype=torch.float32)
    y_train = torch.tensor(y_train_pd.values, dtype=torch.float32)
    X_test = torch.tensor(X_test_pd.values, dtype=torch.float32)
    y_test = torch.tensor(y_test_pd.values, dtype=torch.float32)
    
    # 改变y的形状，使其从 (n,) 变为 (n, 1)，以匹配模型输出
    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)
    
    print("\n已将数据全部转换为PyTorch Tensors。")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

except Exception as e:
    print(f"在分割或转换数据时发生错误: {e}")

from torch.utils.data import TensorDataset, DataLoader

# --- 步骤 2.1: 封装数据到 Dataset 和 DataLoader ---
try:
    print("\n开始封装数据到DataLoader...")
    
    # --- 训练数据加载器 ---
    # 1. 将训练数据打包成TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    
    # 2. 创建DataLoader
    batch_size = 64 # 我们可以设定一个常用的小批量大小，比如64
    # shuffle=True 表示在每个epoch开始时，都会打乱数据顺序
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    
    # --- 测试数据加载器 ---
    # 1. 将测试数据打包成TensorDataset
    test_dataset = TensorDataset(X_test, y_test)
    
    # 2. 创建DataLoader
    # 对于测试集，我们通常不需要打乱顺序 (shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoader创建成功！")
    
    # 我们可以迭代一次，看看它输出的数据是什么样的
    # next(iter(...)) 可以方便地取出第一个批次
    features_batch, labels_batch = next(iter(train_loader))

    
except Exception as e:
    print(f"在创建DataLoader时发生错误: {e}")

import torch.nn as nn

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
    learning_rate = 0.000000001  # 这是一个需要调整的超参数，我们先设一个较小的值
    wd = 0.0000001             # L2正则化(权重衰减)的系数lambda，也是超参数 
    # 使用随机梯度下降(SGD)优化器
    # model.parameters() 会自动将模型所有的可学习参数交给优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd)
    
except Exception as e:
    print(f"在定义模型、损失或优化器时发生错误: {e}")

# --- 步骤 4: 训练模型 ---

# 设定训练轮次
num_epochs = 750

print("\n开始训练模型...")

# 用于记录每个epoch的训练和测试损失
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # --- 训练阶段 ---
    model.train() # 将模型设置为训练模式
    
    current_train_loss = 0.0
    for i, (features, labels) in enumerate(train_loader):
        # 1. 清空梯度
        optimizer.zero_grad()
        
        # 2. 前向传播
        outputs = model(features)
        
        # 3. 计算损失
        loss = loss_fn(outputs, labels)
        
        # 4. 反向传播
        loss.backward()
        
        # 5. 更新参数
        optimizer.step()
        
        # 累加训练损失
        current_train_loss += loss.item()
        
    # 计算平均训练损失
    avg_train_loss = current_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # --- 评估阶段 ---
    model.eval() # 将模型设置为评估模式
    current_test_loss = 0.0
    with torch.no_grad(): # 在此模式下，不计算梯度
        for features, labels in test_loader:
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            current_test_loss += loss.item()
            
    # 计算平均测试损失和RMSE
    avg_test_loss = current_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    rmse = torch.sqrt(torch.tensor(avg_test_loss))
    
    # 打印每个epoch的结果
    # 我们用 \r 来实现动态更新一行的效果
    print(f"\rEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test RMSE: {rmse:.4f}", end="")

    

print("\n\n训练完成！")


import matplotlib.pyplot as plt

# --- 步骤 5: 结果可视化与分析与保存 ---
try:

    print("\n--- 步骤 6: 可视化训练过程 ---")

    plt.figure(figsize=(10, 5))
    start_epoch = 200 # 从第250个epoch开始画
    
    # 创建横坐标，从 start_epoch 到 num_epochs
    epoch_range = range(start_epoch, num_epochs)
    
    # 从损失记录中，只取出我们需要的部分
    train_losses_to_plot = train_losses[start_epoch:]
    test_losses_to_plot = test_losses[start_epoch:]
    
    # 绘图
    plt.plot(epoch_range, train_losses_to_plot, label='Train Loss')
    plt.plot(epoch_range, test_losses_to_plot, label='Test Loss')
    
    plt.xlabel(f'Epoch (starting from {start_epoch})')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Test Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"在可视化或分析结果时发生错误: {e}")

import numpy as np
# ==================================================================
#           在脚本最末尾，添加以下“最佳/最差案例”分析模块
# ==================================================================

print("\n--- 步骤 Y: 分析最佳与最差预测案例 ---")

try:
    # 1. 准备工作：收集测试集的所有预测结果
    model.eval() # 确保模型在评估模式
    all_features = []
    all_true_labels = []
    all_predictions = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            # .cpu()是将数据从可能存在的GPU移回CPU
            # .numpy()是将Tensor转换为NumPy数组，方便后续操作
            all_features.append(features.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            
    # 将批次的列表合并成一个大的NumPy数组
    all_features = np.concatenate(all_features, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0).flatten() # flatten转为一维
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()

    # 2. 计算每个样本的绝对误差
    errors = np.abs(all_predictions - all_true_labels)
    
    # 3. 找到误差最小和最大的5个样本的索引
    # np.argsort会返回排序后的原始索引
    sorted_indices = np.argsort(errors)
    
    worst_indices = sorted_indices[-5:] # 取最后5个，即误差最大的
    best_indices = sorted_indices[:5]  # 取最前5个，即误差最小的

    # 4. 定义一个漂亮的打印函数
    def print_cases(title, indices, features, labels, predictions, errors, feature_names):
        print(f"\n--- {title} ---")
        for i, idx in enumerate(indices):
            print(f"\n案例 {i+1} (原始索引: {idx})")
            print(f"  预测值: {predictions[idx]:.2f}°C, 真实值: {labels[idx]:.2f}°C, 绝对误差: {errors[idx]:.2f}°C")
            print("  参与决策的输入特征:")
            # 将特征值和特征名一一对应打印出来
            for name, value in zip(feature_names, features[idx]):
                print(f"    - {name}: {value:.2f}")
    
    # 获取特征名称列表，用于打印
    feature_names = X_train_pd.columns
    
    # 5. 打印结果
    # 注意：我们传入的是未经标准化的 X_test_pd.values 来展示原始特征
    # 这样更直观。但模型内部处理的是标准化后的数据。
    # 为了简化，我们这里直接展示Tensor转换前的Pandas数据
    print_cases("误差最大的5个预测", worst_indices, X_test_pd.values, y_test_pd.values.flatten(), all_predictions, errors, feature_names)
    print_cases("误差最小的5个预测", best_indices, X_test_pd.values, y_test_pd.values.flatten(), all_predictions, errors, feature_names)


except Exception as e:
    print(f"\n在分析最佳/最差案例时发生错误: {e}")
    import traceback
    traceback.print_exc()
# ==================================================================
#           在你现有代码的最末尾，添加以下模型保存模块
# ==================================================================

print("\n--- 步骤 X: 保存模型参数到CSV/TXT文件 ---")

try:
    # 1. 从训练好的模型中提取参数
    # model.linear.weight 是权重，.data可以获取其数据张量
    # .T 是转置，将其从行向量变为列向量
    weights = model.linear.weight.data.T 
    
    # model.linear.bias 是偏置
    bias = model.linear.bias.data

    print("成功从模型中提取了权重和偏置。")
    print(f"权重的形状: {weights.shape}")
    print(f"偏置的值: {bias.item():.4f}")

    # 2. 将权重和其对应的特征名准备成可保存的格式
    # X_train_pd 是我们分割数据集时得到的Pandas DataFrame，包含了所有特征的列名
    feature_names = X_train_pd.columns
    
    # 创建一个Pandas DataFrame来保存权重
    # 我们把特征名和权重值并排放在一起
    weights_df = pd.DataFrame({
        'feature': feature_names,
        'weight': weights.numpy().flatten() # .numpy()转为numpy数组, .flatten()转为一维
    })

    # 3. 保存到文件
    WEIGHTS_PATH = 'model_weights.csv'
    BIAS_PATH = 'model_bias.txt'

    # 将权重DataFrame保存为CSV文件
    weights_df.to_csv(WEIGHTS_PATH, index=False, encoding='utf-8-sig')
    
    # 将偏置保存为TXT文件
    with open(BIAS_PATH, 'w') as f:
        f.write(str(bias.item()))
        
    print(f"\n权重已保存到可读的CSV文件: {WEIGHTS_PATH}")
    print(f"偏置已保存到可读的TXT文件: {BIAS_PATH}")

except Exception as e:
    print(f"\n在保存参数时发生错误: {e}")