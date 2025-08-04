from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def datasplit(df):
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
        return train_loader, test_loader, X_train_pd, X_test_pd, y_test_pd,X_train
        # return train_loader, test_loader, X_train_pd, y_train_pd, X_test_pd, y_test_pd 这些后续都有用啊

        
    except Exception as e:
        print(f"在创建DataLoader时发生错误: {e}")

