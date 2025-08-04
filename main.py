import project_one.data_processing
import project_one.analysis
import project_one.data_split
import project_one.model
import project_one.save
import project_one.train

def main():
    file_path = 'D:\\python_file\\AI\\datasets\\DData.csv'
    df = project_one.data_processing.Data_process(file_path)
    train_loader, test_loader, X_train_pd, X_test_pd, y_test_pd,X_train = project_one.data_split.datasplit(df)
    # loader的作用是加载数据，X_train_dp 是原pd形式的训练数组，主要用来获取原各个特征的名称。而X_test,y_test_pd主要用于分析结果，X_train获取特征数量
    model,loss_fn,optimizer = project_one.model.define_model(X_train)
    model, train_losses, test_losses = project_one.train.train(model,train_loader,test_loader,loss_fn,optimizer)
    project_one.analysis.show_graph(train_losses,test_losses)
    project_one.analysis.analyze_best_worst_cases(model, test_loader, X_test_pd, y_test_pd, X_train_pd)
    project_one.save.save_model(model,X_train_pd)
    print("Done!")

if __name__ == "__main__":
    main()