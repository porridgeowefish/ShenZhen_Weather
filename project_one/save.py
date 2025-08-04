import pandas as pd
def save_model(model, X_train_pd):
    """
    保存训练好的模型参数到CSV/TXT文件。
    
    :param model: 训练好的PyTorch模型
    """
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