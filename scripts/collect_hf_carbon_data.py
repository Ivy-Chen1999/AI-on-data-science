#!/usr/bin/env python3
"""
🌍 Collect Real Carbon Data from HuggingFace Hub
使用HF官方API获取真实的碳排放数据
"""

from huggingface_hub import HfApi, list_models
import json
import pandas as pd
from pathlib import Path
import time

def collect_carbon_models():
    """收集有碳排放数据的HuggingFace模型"""

    print("🌍 Collecting Models with Carbon Emission Data")
    print("=" * 50)

    api = HfApi()

    # 搜索有碳排放数据的模型
    print("🔍 Searching for models with carbon emission data...")

    models_with_carbon = []

    try:
        # 方法1: 直接搜索有co2_eq_emissions的模型
        models = list(list_models(
            filter="co2_eq_emissions",
            limit=100,
            sort="downloads"
        ))

        print(f"📊 Found {len(models)} models with carbon emission metadata")

        for i, model in enumerate(models):
            if i % 10 == 0:
                print(f"Processing {i}/{len(models)} models...")

            try:
                # 获取模型详细信息
                model_info = api.model_info(model.id)

                # 提取碳排放数据
                carbon_data = {
                    'model_id': model.id,
                    'model_name': model.id.split('/')[-1],
                    'downloads': getattr(model, 'downloads', 0),
                    'likes': getattr(model, 'likes', 0),
                    'tags': getattr(model, 'tags', []),
                    'library_name': getattr(model, 'library_name', 'unknown'),
                    'created_at': getattr(model, 'created_at', None),
                    'last_modified': getattr(model, 'last_modified', None),
                }

                # 检查是否有碳排放数据
                if hasattr(model_info, 'card_data') and model_info.card_data:
                    card_data = model_info.card_data

                    # 提取CO2数据
                    if 'co2_eq_emissions' in card_data:
                        co2_data = card_data['co2_eq_emissions']
                        carbon_data.update({
                            'co2_eq_emissions': co2_data.get('emissions'),
                            'source': co2_data.get('source'),
                            'training_type': co2_data.get('training_type'),
                            'geographical_location': co2_data.get('geographical_location'),
                            'hardware_used': co2_data.get('hardware_used')
                        })

                        models_with_carbon.append(carbon_data)
                        print(f"  ✅ {model.id} | CO2: {co2_data.get('emissions')} | Source: {co2_data.get('source')}")

                # 避免API限制
                time.sleep(0.1)

            except Exception as e:
                print(f"  ⚠️  Error processing {model.id}: {e}")
                continue

    except Exception as e:
        print(f"❌ Error searching models: {e}")
        return []

    return models_with_carbon

def search_popular_models_with_carbon():
    """搜索热门模型中的碳排放数据"""

    print("\n🔥 Searching popular models for carbon data...")

    # 搜索最热门的模型
    popular_models = list(list_models(
        sort="downloads",
        limit=200  # 检查前200个最热门的模型
    ))

    carbon_models = []
    api = HfApi()

    for i, model in enumerate(popular_models):
        if i % 20 == 0:
            print(f"Checking popular model {i}/200...")

        try:
            model_info = api.model_info(model.id)

            # 检查模型卡内容
            if hasattr(model_info, 'card_data') and model_info.card_data:
                card_data = model_info.card_data

                if 'co2_eq_emissions' in card_data:
                    co2_data = card_data['co2_eq_emissions']

                    carbon_info = {
                        'model_id': model.id,
                        'downloads': getattr(model, 'downloads', 0),
                        'co2_emissions': co2_data.get('emissions'),
                        'source': co2_data.get('source'),
                        'hardware': co2_data.get('hardware_used'),
                        'location': co2_data.get('geographical_location')
                    }

                    carbon_models.append(carbon_info)
                    print(f"  🌍 Found: {model.id} | {co2_data.get('emissions')} CO2")

            time.sleep(0.1)  # API限制

        except Exception as e:
            continue

    return carbon_models

def main():
    """主函数"""

    # 收集有碳排放数据的模型
    carbon_models = collect_carbon_models()

    # 搜索热门模型中的碳排放数据
    popular_carbon = search_popular_models_with_carbon()

    # 合并数据
    all_carbon_data = carbon_models + popular_carbon

    if all_carbon_data:
        # 转换为DataFrame
        df = pd.DataFrame(all_carbon_data)

        # 去重
        df = df.drop_duplicates(subset=['model_id'])

        # 保存数据
        output_path = Path("data/raw/huggingface_real_carbon_data.csv")
        df.to_csv(output_path, index=False)

        print(f"\n✅ 收集到 {len(df)} 个有真实碳排放数据的模型")
        print(f"💾 数据已保存到: {output_path}")

        # 统计信息
        print(f"\n📊 数据统计:")
        print(f"  • 有CO2数据的模型: {len(df[df['co2_eq_emissions'].notna()])}")
        print(f"  • 平均下载量: {df['downloads'].mean():,.0f}")
        print(f"  • 数据来源分布:")

        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            for source, count in source_counts.items():
                print(f"    - {source}: {count} 个模型")

        # 显示样本数据
        print(f"\n🔍 样本数据:")
        sample_models = df.head(10)
        for _, model in sample_models.iterrows():
            print(f"  {model['model_id'][:40]:40} | CO2: {model.get('co2_eq_emissions', 'N/A')}")

    else:
        print("\n❌ 未找到有碳排放数据的模型")
        print("💡 建议使用现有的ML CO2方法进行估算")

if __name__ == "__main__":
    main()