import torch
def train(model, train_loader, test_loader, loss_fn, optimizer):
    """
    训练模型的函数
    :param model: 定义好的PyTorch模型
    :param train_loader: 训练数据加载器
    :param test_loader: 测试数据加载器
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    """
    
    # --- 步骤 4: 开始训练 ---
    print("\n开始训练模型...")
    
    # 设定训练轮次
    num_epochs = 750
    
    print("\n开始训练模型...")
    
    # 用于记录每个epoch的训练和测试损失
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()  # 将模型设置为训练模式
        
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
        model.eval()  # 将模型设置为评估模式
        current_test_loss = 0.0
        with torch.no_grad():  # 在此模式下，不计算梯度
            for features, labels in test_loader:
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                current_test_loss += loss.item()
                
        # 计算平均测试损失和RMSE
        avg_test_loss = current_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        rmse = torch.sqrt(torch.tensor(avg_test_loss))
        
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test RMSE: {rmse:.4f}", end="")
    print("\n\n训练完成！")
    return model, train_losses, test_losses