import pandas as pd
import numpy as np
from decimal import Decimal, getcontext

# 設置 Decimal 精度
getcontext().prec = 50

# 常數 (來自提供的程式碼的預設值)
aice = Decimal("0.6")
frac_sc = Decimal("1")
sc = Decimal("1370")
K = Decimal("3.87")
tcrit = Decimal("-10")
B = Decimal("2.17")
A = Decimal("204")
a = Decimal("0.3")

# 從 Excel 檔案中提取的數據
zones = ["80-90", "70-80", "60-70", "50-60", "40-50", "30-40", "20-30", "10-20", "0-10"]
sunwt = [Decimal("0.5"), Decimal("0.531"), Decimal("0.624"), Decimal("0.77"), Decimal("0.892"), Decimal("1.021"), Decimal("1.12"), Decimal("1.189"), Decimal("1.219")]
init_t = [Decimal("-15"), Decimal("-15"), Decimal("-5"), Decimal("5"), Decimal("10"), Decimal("15"), Decimal("18"), Decimal("22"), Decimal("24")]
lat = [Decimal("85"), Decimal("75"), Decimal("65"), Decimal("55"), Decimal("45"), Decimal("35"), Decimal("25"), Decimal("15"), Decimal("6")]
init_a = [aice if t < tcrit else a for t in init_t]  # 初始化 init_a

def energy_balance_model_iterative(zones, sunwt, init_t, init_a, lat, max_iterations=500, tolerance=Decimal("0.0000000001")):
    final_t = [t for t in init_t]
    final_a = [a for a in init_a]
    all_temps = []
    global_mean_temp_final = Decimal("0")
    all_tcos = []

    for iteration in range(max_iterations):
        # 計算 Tcos 和全球平均溫度
        Tcos = [final_t[i] * Decimal(np.cos(np.radians(float(lat[i])))) for i in range(len(zones))]
        mean_temp = sum(Tcos) / sum(Decimal(np.cos(np.radians(float(l)))) for l in lat)
        all_tcos.append(Tcos)

        new_t = []
        new_a = []
        r_in_list = [(sc / Decimal("4")) * frac_sc * sw for sw in sunwt]
        r_out_list = []
        f_i_list = []
        r_in_1_a_list = []
        balance_list = []

        for i in range(len(zones)):
            r_in_i = r_in_list[i]
            r_out_i = A + B * final_t[i]
            
            # 這裡是問題所在：應該使用當前的 albedo，而不是更新後的
            temp_i = (r_in_i * (Decimal("1") - final_a[i]) + K * mean_temp - A) / (B + K)
            albedo_i = aice if temp_i < tcrit else a
            
            new_t.append(temp_i)
            new_a.append(albedo_i)
            r_out_list.append(r_out_i)
            f_i_i = K * (temp_i - mean_temp)
            f_i_list.append(f_i_i)
            r_in_1_a = r_in_i * (Decimal("1") - final_a[i])
            r_in_1_a_list.append(r_in_1_a)
            balance = r_out_i + f_i_i - r_in_1_a
            balance_list.append(balance)

        # 檢查收斂性
        converged = True
        for i in range(len(zones)):
            if abs(new_t[i] - final_t[i]) > tolerance:
                converged = False
                break
        
        final_t = new_t
        final_a = new_a
        all_temps.append(final_t[:])
        global_mean_temp_final = mean_temp
        
        if converged:
            print(f"Converged after {iteration + 1} iterations")
            break

    return final_t, final_a, global_mean_temp_final, all_temps, r_in_list, r_out_list, f_i_list, all_tcos, r_in_1_a_list, balance_list, init_a

def energy_balance_model_corrected(zones, sunwt, init_t, init_a, lat, max_iterations=500, tolerance=Decimal("0.0000000001")):
    """修正版本的能量平衡模型"""
    current_t = [t for t in init_t]
    current_a = [a for a in init_a]
    all_temps = []
    
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
        all_temps.append(current_t[:])
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Max temperature change = {max_change:.10f}")
        
        if converged:
            print(f"Converged after {iteration + 1} iterations with max change = {max_change:.10f}")
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
    
    return current_t, current_a, final_mean_temp, all_temps, r_in_list, r_out_list, f_i_list, [tcos_final], r_in_1_a_list, balance_list, current_a

print("=== 原始版本 ===")
final_t_orig, final_a_orig, global_mean_temp_orig, all_temps_orig, r_in_orig, r_out_orig, f_i_orig, all_tcos_orig, r_in_1_a_list_orig, balance_list_orig, init_a_final_orig = energy_balance_model_iterative(zones, sunwt, init_t, init_a, lat, max_iterations=500, tolerance=Decimal("0.0000000001"))

print("\n=== 修正版本 ===")
final_t_corr, final_a_corr, global_mean_temp_corr, all_temps_corr, r_in_corr, r_out_corr, f_i_corr, all_tcos_corr, r_in_1_a_list_corr, balance_list_corr, init_a_final_corr = energy_balance_model_corrected(zones, sunwt, init_t, init_a, lat, max_iterations=500, tolerance=Decimal("0.0000000001"))

# 比較結果
print("\n=== 結果比較 ===")
comparison_df = pd.DataFrame({
    'Zone': zones,
    'Original_T': [float(t) for t in final_t_orig],
    'Corrected_T': [float(t) for t in final_t_corr],
    'Difference': [float(final_t_corr[i] - final_t_orig[i]) for i in range(len(zones))],
    'Original_Balance': [float(b) for b in balance_list_orig],
    'Corrected_Balance': [float(b) for b in balance_list_corr]
})

print(comparison_df)
print(f"\n原始全球平均溫度: {float(global_mean_temp_orig):.6f}°C")
print(f"修正全球平均溫度: {float(global_mean_temp_corr):.6f}°C")
print(f"差異: {float(global_mean_temp_corr - global_mean_temp_orig):.6f}°C")
