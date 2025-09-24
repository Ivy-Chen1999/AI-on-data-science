#!/usr/bin/env python3
"""
📊 比较纯物理公式 vs 混合预测系统的性能
使用CodeCarbon真实数据作为ground truth
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def compare_prediction_methods():
    """比较不同预测方法的准确性"""

    print("🔬 纯物理公式 vs 混合预测系统 比较分析")
    print("=" * 60)

    # 1. 加载CodeCarbon真实数据
    data_dir = Path("data/raw")
    codecarbon_df = pd.read_csv(data_dir / "high_quality_carbon_data.csv")

    # 过滤极端值
    codecarbon_df = codecarbon_df[codecarbon_df['co2_kg'] < 50000]

    print(f"📊 分析 {len(codecarbon_df)} 个CodeCarbon真实测量数据")

    # 2. 加载各种预测结果
    predictions = []

    # 检查是否有各种预测结果文件
    if (data_dir / "calibration_data.csv").exists():
        calib_df = pd.read_csv(data_dir / "calibration_data.csv")
        print(f"✅ 加载校准数据: {len(calib_df)} 个样本")

        # 合并数据
        for _, row in calib_df.iterrows():
            predictions.append({
                'model_name': row['model_name'],
                'real_co2_kg': row['real_co2_kg'],
                'physical_baseline_kg': row['basic_physical_kg'],
                'calibration_ratio': row['calibration_ratio']
            })

    # 如果有实际验证结果
    if (data_dir / "practical_validation_results.csv").exists():
        val_df = pd.read_csv(data_dir / "practical_validation_results.csv")

        # 更新predictions数据
        for _, row in val_df.iterrows():
            found = False
            for pred in predictions:
                if pred['model_name'] == row['model_name']:
                    pred['practical_corrected_kg'] = row['corrected_pred_kg']
                    found = True
                    break

            if not found:
                predictions.append({
                    'model_name': row['model_name'],
                    'real_co2_kg': row['real_co2_kg'],
                    'physical_baseline_kg': row['physical_pred_kg'],
                    'practical_corrected_kg': row['corrected_pred_kg']
                })

    if not predictions:
        print("⚠️ 没有找到预测数据，使用原始CodeCarbon数据估算")
        # 简单估算
        for _, row in codecarbon_df.iterrows():
            # 假设的物理baseline (非常粗略)
            physical_estimate = 0.1  # kg CO2 (极度低估)
            predictions.append({
                'model_name': row['model_name'],
                'real_co2_kg': row['co2_kg'],
                'physical_baseline_kg': physical_estimate
            })

    pred_df = pd.DataFrame(predictions)

    # 3. 计算误差指标
    print("\n📈 预测性能比较:")
    print("-" * 60)

    # 纯物理公式的误差
    if 'physical_baseline_kg' in pred_df.columns:
        physical_errors = np.abs(pred_df['physical_baseline_kg'] - pred_df['real_co2_kg']) / pred_df['real_co2_kg'] * 100
        physical_mae = np.mean(np.abs(pred_df['physical_baseline_kg'] - pred_df['real_co2_kg']))
        physical_median_error = np.median(physical_errors)
        physical_mean_error = np.mean(physical_errors)

        # 计算方向性偏差
        underestimate_ratio = sum(pred_df['physical_baseline_kg'] < pred_df['real_co2_kg']) / len(pred_df) * 100

        print("\n🔵 纯物理公式 (ML CO2 baseline):")
        print(f"  平均绝对误差: {physical_mae:.2f} kg CO₂")
        print(f"  平均相对误差: {physical_mean_error:.1f}%")
        print(f"  中位数相对误差: {physical_median_error:.1f}%")
        print(f"  低估比例: {underestimate_ratio:.1f}% (物理公式倾向于低估)")

        # 误差分布
        error_ranges = [
            (0, 50, "优秀 (<50%)"),
            (50, 100, "良好 (50-100%)"),
            (100, 200, "一般 (100-200%)"),
            (200, 500, "较差 (200-500%)"),
            (500, float('inf'), "很差 (>500%)")
        ]

        print("\n  误差分布:")
        for low, high, label in error_ranges:
            count = sum((physical_errors >= low) & (physical_errors < high))
            pct = count / len(physical_errors) * 100
            if count > 0:
                print(f"    {label}: {count} 个 ({pct:.1f}%)")

    # 混合预测的误差（如果有）
    if 'practical_corrected_kg' in pred_df.columns:
        corrected_errors = np.abs(pred_df['practical_corrected_kg'] - pred_df['real_co2_kg']) / pred_df['real_co2_kg'] * 100
        corrected_mae = np.mean(np.abs(pred_df['practical_corrected_kg'] - pred_df['real_co2_kg']))
        corrected_median_error = np.median(corrected_errors)
        corrected_mean_error = np.mean(corrected_errors)

        print("\n🟢 混合预测系统 (物理+ML校正):")
        print(f"  平均绝对误差: {corrected_mae:.2f} kg CO₂")
        print(f"  平均相对误差: {corrected_mean_error:.1f}%")
        print(f"  中位数相对误差: {corrected_median_error:.1f}%")

        print("\n  误差分布:")
        for low, high, label in error_ranges:
            count = sum((corrected_errors >= low) & (corrected_errors < high))
            pct = count / len(corrected_errors) * 100
            if count > 0:
                print(f"    {label}: {count} 个 ({pct:.1f}%)")

        # 改进对比
        print("\n📊 改进效果:")
        improvement_mae = physical_mae - corrected_mae
        improvement_mean = physical_mean_error - corrected_mean_error
        improvement_median = physical_median_error - corrected_median_error

        print(f"  MAE改进: {improvement_mae:.2f} kg (改进 {improvement_mae/physical_mae*100:.1f}%)")
        print(f"  平均误差改进: {improvement_mean:.1f} 个百分点")
        print(f"  中位数误差改进: {improvement_median:.1f} 个百分点")

    # 4. 具体案例分析
    print("\n🔍 具体案例对比 (前10个):")
    print("-" * 60)

    for i, row in pred_df.head(10).iterrows():
        print(f"\n{row['model_name'][:40]}:")
        print(f"  真实值: {row['real_co2_kg']:.3f} kg CO₂")

        if 'physical_baseline_kg' in row:
            physical_error = abs(row['physical_baseline_kg'] - row['real_co2_kg']) / row['real_co2_kg'] * 100
            print(f"  纯物理: {row['physical_baseline_kg']:.3f} kg (误差 {physical_error:.1f}%)")

        if 'practical_corrected_kg' in row and pd.notna(row['practical_corrected_kg']):
            corrected_error = abs(row['practical_corrected_kg'] - row['real_co2_kg']) / row['real_co2_kg'] * 100
            print(f"  混合预测: {row['practical_corrected_kg']:.3f} kg (误差 {corrected_error:.1f}%)")

            if 'physical_baseline_kg' in row:
                if corrected_error < physical_error:
                    print(f"  ✅ 混合预测更准确 (改进 {physical_error - corrected_error:.1f} 个百分点)")
                else:
                    print(f"  ❌ 纯物理更准确")

    # 5. 统计分析
    print("\n📊 统计分析:")
    print("-" * 60)

    if 'physical_baseline_kg' in pred_df.columns:
        # 计算相关性
        from scipy import stats

        # Log-log相关性（因为数据跨度很大）
        valid_mask = (pred_df['physical_baseline_kg'] > 0) & (pred_df['real_co2_kg'] > 0)
        if sum(valid_mask) > 2:
            log_physical = np.log10(pred_df[valid_mask]['physical_baseline_kg'])
            log_real = np.log10(pred_df[valid_mask]['real_co2_kg'])
            correlation, p_value = stats.pearsonr(log_physical, log_real)

            print(f"  物理预测与真实值的log-log相关性: {correlation:.3f} (p={p_value:.3e})")

            if correlation > 0.7:
                print("  ✅ 强相关: 物理公式捕获了主要趋势")
            elif correlation > 0.4:
                print("  ⚠️ 中等相关: 物理公式部分有效")
            else:
                print("  ❌ 弱相关: 物理公式需要大幅改进")

        # 系统性偏差分析
        bias_ratio = pred_df['physical_baseline_kg'] / pred_df['real_co2_kg']
        geometric_mean_bias = np.exp(np.mean(np.log(bias_ratio[bias_ratio > 0])))

        print(f"\n  系统性偏差分析:")
        print(f"    几何平均偏差因子: {geometric_mean_bias:.2f}x")
        if geometric_mean_bias < 1:
            print(f"    物理公式系统性低估 {(1/geometric_mean_bias):.1f} 倍")
        else:
            print(f"    物理公式系统性高估 {geometric_mean_bias:.1f} 倍")

    # 6. 结论
    print("\n💡 结论:")
    print("-" * 60)

    if 'physical_baseline_kg' in pred_df.columns and 'practical_corrected_kg' in pred_df.columns:
        if corrected_mean_error < physical_mean_error:
            print("✅ 混合预测系统显著优于纯物理公式")
            print(f"   平均误差降低: {physical_mean_error - corrected_mean_error:.1f} 个百分点")
        else:
            print("⚠️ 混合预测未能改进纯物理公式")
            print("   可能原因: 训练数据不足或模型过拟合")
    elif 'physical_baseline_kg' in pred_df.columns:
        if physical_mean_error > 100:
            print("❌ 纯物理公式误差过大 (>100%)，需要校正")
            print("   建议: 使用更多真实数据训练ML校正模型")
        else:
            print("✅ 纯物理公式表现尚可，但仍有改进空间")

    # 7. 建议
    print("\n🚀 改进建议:")
    print("-" * 60)

    if 'physical_baseline_kg' in pred_df.columns:
        if underestimate_ratio > 80:
            print("1. 物理公式严重低估，需要调整基础参数:")
            print("   - 增加PUE值 (当前可能过低)")
            print("   - 调整GPU利用率估计")
            print("   - 考虑额外的overhead (数据加载、评估等)")
        elif underestimate_ratio < 20:
            print("1. 物理公式可能高估，检查:")
            print("   - GPU功耗是否过高")
            print("   - 训练时间估计是否准确")

    print("\n2. 数据收集:")
    print("   - 继续收集更多CodeCarbon测量数据")
    print("   - 特别关注目前预测误差大的模型类型")
    print("   - 记录更详细的硬件和训练配置信息")

    print("\n3. 模型改进:")
    print("   - 为不同模型类型训练专门的校正模型")
    print("   - 考虑非线性校正因子")
    print("   - 引入更多特征 (数据集大小、批次大小等)")

    # 保存分析结果
    result_summary = {
        'n_samples': len(pred_df),
        'physical_mean_error': physical_mean_error if 'physical_baseline_kg' in pred_df.columns else None,
        'physical_median_error': physical_median_error if 'physical_baseline_kg' in pred_df.columns else None,
        'corrected_mean_error': corrected_mean_error if 'practical_corrected_kg' in pred_df.columns else None,
        'corrected_median_error': corrected_median_error if 'practical_corrected_kg' in pred_df.columns else None,
        'underestimate_ratio': underestimate_ratio if 'physical_baseline_kg' in pred_df.columns else None
    }

    return result_summary

if __name__ == "__main__":
    results = compare_prediction_methods()

    print("\n" + "=" * 60)
    print("📊 分析完成!")
    print("=" * 60)