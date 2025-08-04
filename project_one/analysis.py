import matplotlib.pyplot as plt
import numpy as np
import torch 
# --- 步骤 5: 结果可视化与分析与保存 ---
def show_graph(train_losses, test_losses, num_epochs=750):
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
# ==================================================================
#           在脚本最末尾，添加以下“最佳/最差案例”分析模块
# ==================================================================

def analyze_best_worst_cases(model, test_loader, X_test_pd, y_test_pd, X_train_pd):
    """
    分析模型在测试集上的最佳和最差预测案例。
    
    :param model: 训练好的模型
    """

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