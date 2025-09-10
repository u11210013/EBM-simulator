import pandas as pd
import numpy as np
from decimal import Decimal, getcontext

# 設置 Decimal 精度
getcontext().prec = 50

# 常數
aice = Decimal("0.6")
frac_sc = Decimal("1")
sc = Decimal("1370")
K = Decimal("3.87")
tcrit = Decimal("-10")
B = Decimal("2.17")
A = Decimal("204")
a = Decimal("0.3")

# 數據
zones = ["80-90", "70-80", "60-70", "50-60", "40-50", "30-40", "20-30", "10-20", "0-10"]
sunwt = [Decimal("0.5"), Decimal("0.531"), Decimal("0.624"), Decimal("0.77"), Decimal("0.892"), Decimal("1.021"), Decimal("1.12"), Decimal("1.189"), Decimal("1.219")]
init_t = [Decimal("-15"), Decimal("-15"), Decimal("-5"), Decimal("5"), Decimal("10"), Decimal("15"), Decimal("18"), Decimal("22"), Decimal("24")]
lat = [Decimal("85"), Decimal("75"), Decimal("65"), Decimal("55"), Decimal("45"), Decimal("35"), Decimal("25"), Decimal("15"), Decimal("6")]

def energy_balance_model_corrected(zones, sunwt, init_t, lat, max_iterations=500, tolerance=Decimal("0.001")):
    """修正版本的能量平衡模型"""
    
    # 初始化
    current_t = [t for t in init_t]
    current_a = [aice if t < tcrit else a for t in init_t]
    all_temps = []
    
    print("開始迭代計算...")
    
    for iteration in range(max_iterations):
        # 計算加權全球平均溫度
        cos_weights = [Decimal(np.cos(np.radians(float(l)))) for l in lat]
        weighted_temp_sum = sum(current_t[i] * cos_weights[i] for i in range(len(zones)))
        total_weight = sum(cos_weights)
        mean_temp = weighted_temp_sum / total_weight
        
        new_t = []
        new_a = []
        
        # 計算每個區域的新溫度
        for i in range(len(zones)):
            # 入射太陽輻射
            r_in = (sc / Decimal("4")) * frac_sc * sunwt[i]
            
            # 能量平衡方程：R_in * (1-a) + K * (T_global - T_i) = A + B * T_i
            # 重新排列：T_i = (R_in * (1-a) + K * T_global - A) / (B + K)
            temp_i = (r_in * (Decimal("1") - current_a[i]) + K * mean_temp - A) / (B + K)
            
            # 根據溫度更新反照率
            albedo_i = aice if temp_i < tcrit else a
            
            new_t.append(temp_i)
            new_a.append(albedo_i)
        
        # 檢查收斂性
        converged = True
        max_change = Decimal("0")
        for i in range(len(zones)):
            change = abs(new_t[i] - current_t[i])
            if change > max_change:
                max_change = change
            if change > tolerance:
                converged = False
        
        current_t = new_t
        current_a = new_a
        all_temps.append([float(t) for t in current_t])
        
        if iteration % 100 == 0 or iteration < 10:
            print(f"迭代 {iteration + 1}: 最大溫度變化 = {float(max_change):.8f}°C")
        
        if converged:
            print(f"在第 {iteration + 1} 次迭代後收斂，最大變化 = {float(max_change):.8f}°C")
            break
    
    # 計算最終結果
    cos_weights = [Decimal(np.cos(np.radians(float(l)))) for l in lat]
    weighted_temp_sum = sum(current_t[i] * cos_weights[i] for i in range(len(zones)))
    total_weight = sum(cos_weights)
    final_mean_temp = weighted_temp_sum / total_weight
    
    # 計算其他輸出變量
    r_in_list = [(sc / Decimal("4")) * frac_sc * sw for sw in sunwt]
    r_out_list = [A + B * t for t in current_t]
    f_i_list = [K * (t - final_mean_temp) for t in current_t]
    r_in_1_a_list = [r_in_list[i] * (Decimal("1") - current_a[i]) for i in range(len(zones))]
    balance_list = [r_out_list[i] + f_i_list[i] - r_in_1_a_list[i] for i in range(len(zones))]
    tcos_final = [current_t[i] * cos_weights[i] for i in range(len(zones))]
    
    return current_t, current_a, final_mean_temp, all_temps, r_in_list, r_out_list, f_i_list, tcos_final, r_in_1_a_list, balance_list

# 執行修正版本
print("=== 修正版能量平衡模型 ===")
final_t, final_a, global_mean_temp, all_temps, r_in, r_out, f_i, tcos, r_in_1_a_list, balance_list = energy_balance_model_corrected(
    zones, sunwt, init_t, lat, max_iterations=500, tolerance=Decimal("0.001")
)

# 創建結果 DataFrame
results_df = pd.DataFrame({
    'Zones': zones,
    'Lat': [float(l) for l in lat],
    'SunWt': [float(sw) for sw in sunwt],
    'Init_T': [float(t) for t in init_t],
    'Final_T': [float(t) for t in final_t],
    'Final_a': [float(a) for a in final_a],
    'R_in': [float(r) for r in r_in],
    'R_out': [float(r) for r in r_out],
    'R_in*(1-a)': [float(r) for r in r_in_1_a_list],
    'Balance': [float(b) for b in balance_list],
    'F(i)': [float(f) for f in f_i],
    'Tcos': [float(t) for t in tcos]
})

print("\n=== 最終結果 ===")
print(results_df.round(6))
print(f"\n全球平均溫度: {float(global_mean_temp):.6f}°C")

# 檢查能量平衡
print(f"\n=== 能量平衡檢查 ===")
max_imbalance = max([abs(float(b)) for b in balance_list])
print(f"最大能量不平衡: {max_imbalance:.10f}")

if max_imbalance < 0.001:
    print("✅ 能量平衡達到收斂！")
else:
    print("❌ 能量平衡未完全收斂")

# 顯示溫度範圍
temps = [float(t) for t in final_t]
print(f"\n溫度範圍: {min(temps):.2f}°C 到 {max(temps):.2f}°C")
print(f"溫度差: {max(temps) - min(temps):.2f}°C")
