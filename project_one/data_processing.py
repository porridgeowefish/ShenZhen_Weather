import pandas as pd
from lunardate import LunarDate
def Data_process(file_path):
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
        return df    

    except Exception as e:
        print(f"在季节转换或独热编码时发生错误: {e}")


