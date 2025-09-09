#!/usr/bin/env python3
"""
測試熱力圖功能的腳本
這個腳本展示EBM模型的熱力圖可視化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ebm import energy_balance_model_iterative

def test_heatmap_visualization():
    """
    測試熱力圖可視化功能
    """
    print("正在測試EBM熱力圖可視化功能...")
    
    # 設置測試參數
    n_zones = 8
    albedo = 0.3
    solar_constant = 1361
    emissivity = 0.6
    sigma = 5.67e-8
    n_iter = 200
    
    print(f"測試參數:")
    print(f"  區域數量: {n_zones}")
    print(f"  反照率: {albedo}")
    print(f"  太陽常數: {solar_constant} W/m²")
    print(f"  發射率: {emissivity}")
    print(f"  迭代次數: {n_iter}")
    
    # 運行EBM模型
    print("\n正在運行EBM模型...")
    zone_temps_history, global_avg_temps = energy_balance_model_iterative(
        n_zones, albedo, solar_constant, emissivity, sigma, n_iter
    )
    
    print("模型運行完成！")
    print(f"最終全球平均溫度: {global_avg_temps[-1]:.2f} K ({global_avg_temps[-1]-273.15:.2f} °C)")
    
    # 創建多個可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('EBM模型可視化測試 - 熱力圖功能', fontsize=16, fontweight='bold')
    
    # 1. 收斂曲線
    ax1 = axes[0, 0]
    iterations = range(n_iter)
    for i in range(n_zones):
        ax1.plot(iterations, zone_temps_history[:, i], label=f'Zone {i+1}', alpha=0.7)
    ax1.plot(iterations, global_avg_temps, 'k--', linewidth=2, label='Global Average')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('收斂曲線 (Convergence Curve)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 最終溫度長條圖
    ax2 = axes[0, 1]
    zone_names = [f"Zone {i+1}" for i in range(n_zones)]
    final_temps = zone_temps_history[-1, :]
    bars = ax2.bar(zone_names, final_temps, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # 添加數值標籤
    for bar, temp in zip(bars, final_temps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{temp:.1f}K\n({temp-273.15:.1f}°C)',
               ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('最終溫度長條圖 (Final Temperature Bar Chart)')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # 3. 熱力圖 - 使用imshow
    ax3 = axes[1, 0]
    heatmap_data = zone_temps_history.T  # 轉置，使區域為行，迭代為列
    
    im = ax3.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Zone')
    ax3.set_title('溫度熱力圖 (Temperature Heatmap)')
    
    # 設置刻度標籤
    ax3.set_xticks(range(0, n_iter, max(1, n_iter//10)))
    ax3.set_xticklabels(range(0, n_iter, max(1, n_iter//10)))
    ax3.set_yticks(range(n_zones))
    ax3.set_yticklabels([f'Zone {i+1}' for i in range(n_zones)])
    
    # 添加顏色條
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Temperature (K)')
    
    # 4. 熱力圖 - 使用seaborn
    ax4 = axes[1, 1]
    
    # 準備seaborn熱力圖數據
    # 選擇每10次迭代的數據點以減少密度
    step = max(1, n_iter // 20)
    heatmap_data_seaborn = heatmap_data[:, ::step]
    iteration_labels = [str(i) for i in range(0, n_iter, step)]
    
    sns.heatmap(heatmap_data_seaborn, 
                xticklabels=iteration_labels,
                yticklabels=[f'Zone {i+1}' for i in range(n_zones)],
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Temperature (K)'},
                ax=ax4)
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Zone')
    ax4.set_title('溫度熱力圖 (Seaborn Heatmap)')
    
    plt.tight_layout()
    plt.show()
    
    # 顯示統計信息
    print("\n=== 模型結果統計 ===")
    print(f"初始溫度: {zone_temps_history[0, 0]:.2f} K")
    print(f"最終溫度範圍: {np.min(final_temps):.2f} - {np.max(final_temps):.2f} K")
    print(f"溫度變化範圍: {np.max(final_temps) - np.min(final_temps):.2f} K")
    print(f"收斂性檢查: 最後10次迭代的溫度變化 < 0.01 K")
    
    # 檢查收斂性
    last_10_temps = global_avg_temps[-10:]
    temp_variation = np.max(last_10_temps) - np.min(last_10_temps)
    print(f"最後10次迭代的全球平均溫度變化: {temp_variation:.4f} K")
    
    if temp_variation < 0.01:
        print("✅ 模型已收斂！")
    else:
        print("⚠️  模型可能需要更多迭代來收斂")
    
    return zone_temps_history, global_avg_temps

def compare_different_parameters():
    """
    比較不同參數設置下的熱力圖
    """
    print("\n" + "="*50)
    print("比較不同參數設置的熱力圖")
    print("="*50)
    
    # 測試不同的反照率值
    albedo_values = [0.2, 0.3, 0.4]
    n_zones = 5
    n_iter = 100
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('不同反照率下的溫度熱力圖比較', fontsize=16, fontweight='bold')
    
    for idx, albedo in enumerate(albedo_values):
        print(f"\n測試反照率: {albedo}")
        
        # 運行模型
        zone_temps_history, global_avg_temps = energy_balance_model_iterative(
            n_zones, albedo, 1361, 0.6, 5.67e-8, n_iter
        )
        
        # 創建熱力圖
        ax = axes[idx]
        heatmap_data = zone_temps_history.T
        
        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Zone')
        ax.set_title(f'反照率 = {albedo}\n最終溫度: {global_avg_temps[-1]:.1f} K')
        
        # 設置刻度
        ax.set_xticks(range(0, n_iter, max(1, n_iter//10)))
        ax.set_xticklabels(range(0, n_iter, max(1, n_iter//10)))
        ax.set_yticks(range(n_zones))
        ax.set_yticklabels([f'Zone {i+1}' for i in range(n_zones)])
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature (K)')
        
        print(f"  最終全球平均溫度: {global_avg_temps[-1]:.2f} K ({global_avg_temps[-1]-273.15:.2f} °C)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("EBM熱力圖功能測試")
    print("="*50)
    
    # 測試基本熱力圖功能
    zone_temps_history, global_avg_temps = test_heatmap_visualization()
    
    # 比較不同參數
    compare_different_parameters()
    
    print("\n" + "="*50)
    print("測試完成！熱力圖功能運行正常。")
    print("您現在可以在Gradio應用中使用熱力圖可視化功能。")
    print("="*50)
