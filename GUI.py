# ==================================================================
#            最终版GUI应用 V4 (带计算过程展示)
# ==================================================================
import tkinter as tk
from tkinter import messagebox, ttk
from tkcalendar import DateEntry
import pandas as pd
from lunardate import LunarDate

# --- 1. 定义与训练时完全一致的数据加载和预处理函数 ---
def load_and_preprocess_data(file_path):
    # ... (与之前版本完全相同的函数) ...
    column_names = [
        'cumulative_rainfall', 'avg_temp', 'max_humidity', 'min_humidity',
        'avg_pressure', 'min_pressure', 'max_pressure', 'avg_humidity_alt',
        'min_temp', 'max_temp', 'date'
    ]
    df = pd.read_csv(file_path, header=0, names=column_names)
    cols_to_numeric = df.columns.drop('date'); df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
    df['date'] = df['date'].astype(str); df['cumulative_rainfall'].fillna(0, inplace=True); df.dropna(inplace=True)
    cols_to_divide = ['cumulative_rainfall','avg_temp', 'avg_pressure', 'min_pressure', 'max_pressure', 'min_temp', 'max_temp']
    df[cols_to_divide] = df[cols_to_divide] / 10.0
    df['date_dt'] = pd.to_datetime((df['date'].astype('int64') // 1000000).astype(str), format='%Y%m%d')
    return df

# --- 2. 加载模型参数和数据 ---
try:
    WEIGHTS_PATH = 'model_weights.csv'; BIAS_PATH = 'model_bias.txt'
    DATA_PATH = 'D:\\python_file\\AI\\datasets\\DData.csv'
    weights_df = pd.read_csv(WEIGHTS_PATH)
    with open(BIAS_PATH, 'r') as f: bias_value = float(f.read())
    weights_dict = pd.Series(weights_df.weight.values, index=weights_df.feature).to_dict()
    full_df = load_and_preprocess_data(DATA_PATH)
    mean_pressure_train = full_df['avg_pressure'].mean()
    std_pressure_train = full_df['avg_pressure'].std()
except Exception as e:
    messagebox.showerror("加载错误", f"无法加载文件！\n错误: {e}"); exit()

# --- 3. 定义特征工程函数 ---
def feature_engineer(data_row):
    # ... (与之前版本完全相同的函数) ...
    data_row['pressure_diff'] = data_row['max_pressure'] - data_row['min_pressure']
    data_row['pressure_deviation'] = (data_row['avg_pressure'] - mean_pressure_train) / std_pressure_train
    def get_season(date_obj):
        date_only = date_obj.date()
        year = date_only.year
        try:
            lichun=LunarDate(year,1,1).toSolarDate();lixia=LunarDate(year,4,1).toSolarDate()
            liqiu=LunarDate(year,7,1).toSolarDate();lidong=LunarDate(year,10,1).toSolarDate()
            if date_only < lichun: return 'winter'
            elif lichun <= date_only < lixia: return 'spring'
            elif lixia <= date_only < liqiu: return 'summer'
            elif liqiu <= date_only < lidong: return 'autumn'
            else: return 'winter'
        except: return 'unknown'
    season = get_season(data_row['date_dt'])
    seasons = ['spring', 'summer', 'autumn', 'winter']
    for s in seasons: data_row[f'is_{s}'] = 1 if season == s else 0
    return data_row

# --- 4. 定义GUI交互函数 ---
def on_predict():
    try:
        # 清空之前的计算详情
        details_text.delete('1.0', tk.END)

        selected_date = cal.get_date()
        target_row = full_df[full_df['date_dt'].dt.date == selected_date]
        if target_row.empty:
            messagebox.showwarning("无数据", f"找不到 {selected_date.strftime('%Y-%m-%d')} 的数据。")
            return
        data_row = target_row.iloc[0].copy()
        
        info_label.config(text=f"日期: {selected_date.strftime('%Y-%m-%d')}\n平均气温: {data_row['avg_temp']:.1f}°C\n平均气压: {data_row['avg_pressure']:.1f} hPa\n平均湿度: {data_row['avg_humidity_alt']}%")
        
        features_row = feature_engineer(data_row)
        
        # --- 核心改动：记录并展示计算过程 ---
        prediction = 0.0
        calculation_details = [] # 用于存储每一步计算的字符串
        
        # 遍历我们加载的权重字典
        for feature_name, weight in weights_dict.items():
            if feature_name in features_row:
                feature_value = float(features_row[feature_name])
                contribution = feature_value * weight
                prediction += contribution
                
                # 创建详情字符串并添加到列表中
                detail_str = f"{feature_name:<20}: {feature_value:8.2f} * {weight:8.4f} = {contribution:8.2f}"
                calculation_details.append(detail_str)

        # 添加偏置项
        prediction += bias_value
        calculation_details.append("-" * 45)
        calculation_details.append(f"{'Bias (偏置项)':<20}: {'':>8} + {'':>8}   {bias_value:8.2f}")
        calculation_details.append("=" * 45)
        calculation_details.append(f"{'Final Prediction':<20}: {'':>8}   {'':>8} = {prediction:8.2f}")

        # 将详情列表合并成一个字符串，并显示在Text控件中
        details_text.insert(tk.END, "\n".join(calculation_details))
        
        prediction_label.config(text=f"预测最高温度: {prediction:.2f}°C")
        
        global true_temp
        true_temp = data_row['max_temp']
        result_button.config(state=tk.NORMAL)
        error_label.config(text="")
    except Exception as e:
        messagebox.showerror("预测错误", f"在预测过程中发生错误: {e}")

def on_show_result():
    # ... (此函数不变) ...
    if true_temp is not None:
        predicted_temp_text = prediction_label.cget("text")
        predicted_temp = float(predicted_temp_text.split(':')[1].strip().split('°C')[0])
        error = predicted_temp - true_temp
        error_label.config(text=f"真实最高温度: {true_temp:.2f}°C\n预测误差: {error:.2f}°C")

# --- 5. 创建GUI窗口 (新增了详情展示区域) ---
root = tk.Tk()
root.title("天气预测器"); root.geometry("550x650") # 窗口加大以容纳详情
main_frame = ttk.Frame(root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)

# Top frame for controls
top_frame = ttk.Frame(main_frame); top_frame.pack(fill=tk.X, pady=5)
ttk.Label(top_frame, text="请选择日期:").pack(side=tk.LEFT, padx=5)
cal = DateEntry(top_frame, width=12, year=2019, month=6, day=11); cal.pack(side=tk.LEFT, padx=5)
predict_button = ttk.Button(top_frame, text="预测", command=on_predict); predict_button.pack(side=tk.LEFT, padx=10)

# Frame for results
result_frame = ttk.Frame(main_frame, padding="10", relief=tk.RIDGE)
result_frame.pack(fill=tk.X, pady=10)
info_label = ttk.Label(result_frame, text="待查询...", justify=tk.LEFT); info_label.pack(anchor='w')
prediction_label = ttk.Label(result_frame, text="预测最高温度: --.--°C", font=("Helvetica", 14, "bold")); prediction_label.pack(pady=5)
result_button = ttk.Button(result_frame, text="查看结果", command=on_show_result, state=tk.DISABLED); result_button.pack()
error_label = ttk.Label(result_frame, text="", font=("Helvetica", 12), foreground="red"); error_label.pack(pady=5)

# --- 新增：计算详情展示区域 ---
details_frame = ttk.LabelFrame(main_frame, text="计算过程详情 (特征值 * 权重 = 贡献)", padding="10")
details_frame.pack(fill=tk.BOTH, expand=True, pady=10)
details_text = tk.Text(details_frame, wrap=tk.WORD, height=15, font=("Courier New", 9))
details_text.pack(fill=tk.BOTH, expand=True)

true_temp = None
root.mainloop()