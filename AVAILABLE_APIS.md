# 🌐 可用的AI碳排放计算API和工具

## 1. CodeCarbon (开源库) ⭐⭐⭐⭐⭐
**网址**: https://github.com/mlco2/codecarbon

### 特点
- **实时测量**: 直接测量你的训练过程
- **零配置**: pip install即可使用
- **精度最高**: 实际测量，非估算

### 使用方法
```python
pip install codecarbon

from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()
# 训练代码
emissions = tracker.stop()
```

### API限制
- 无限制（本地运行）
- 完全免费

---

## 2. ML CO2 Impact Calculator ⭐⭐⭐⭐
**网址**: https://mlco2.github.io/impact/

### 特点
- **Web界面**: 无需编程
- **考虑地理位置**: 不同国家碳强度
- **硬件数据库**: 45种GPU数据

### API接口
```python
# 可以爬取或模拟表单提交
import requests

data = {
    "hardware": "Tesla V100",
    "hours": "100",
    "provider": "AWS",
    "region": "us-east-1"
}
# POST到他们的计算endpoint
```

### 限制
- 无官方API，需要网页抓取
- 免费使用

---

## 3. Green Algorithms ⭐⭐⭐⭐
**网址**: http://www.green-algorithms.org/

### 特点
- **学术背景**: 剑桥大学开发
- **详细参数**: CPU/GPU/内存都考虑
- **碳补偿建议**: 提供种树数量等

### 使用方法
- Web界面计算器
- 可以通过爬虫获取结果

### 限制
- 无官方API
- 需要详细硬件信息

---

## 4. Climatiq API ⭐⭐⭐⭐
**网址**: https://www.climatiq.io/

### 特点
- **商业API**: 专业碳排放计算
- **广泛覆盖**: 不仅限于AI
- **实时数据**: 包含最新碳强度

### API使用
```python
import requests

headers = {
    "Authorization": "Bearer YOUR_API_KEY"
}

data = {
    "emission_factor": "electricity-energy_source_grid_mix",
    "parameters": {
        "energy": 100,
        "energy_unit": "kWh"
    },
    "region": "US"
}

response = requests.post(
    "https://beta3.api.climatiq.io/estimate",
    headers=headers,
    json=data
)
```

### 定价
- **免费**: 250次/月
- **付费**: $49起/月

---

## 5. ElectricityMap API ⭐⭐⭐
**网址**: https://app.electricitymaps.com/

### 特点
- **实时碳强度**: 全球电网实时数据
- **精确到小时**: 考虑时段差异
- **可视化地图**: 直观展示

### API使用
```python
# 需要API key
import requests

headers = {
    "auth-token": "YOUR_TOKEN"
}

response = requests.get(
    "https://api.electricitymap.org/v3/carbon-intensity/latest",
    params={"zone": "US-CAL-CISO"},
    headers=headers
)
```

### 定价
- **个人**: €15/月
- **商业**: 联系定价

---

## 6. WattTime API ⭐⭐⭐
**网址**: https://www.watttime.org/

### 特点
- **实时+预测**: 未来碳强度预测
- **优化建议**: 最佳训练时间
- **非营利组织**: 使命驱动

### API使用
```python
import requests
from requests.auth import HTTPBasicAuth

# 登录获取token
login = requests.get(
    "https://api2.watttime.org/v2/login",
    auth=HTTPBasicAuth('username', 'password')
)
token = login.json()['token']

# 获取碳强度
headers = {'Authorization': f'Bearer {token}'}
data = requests.get(
    "https://api2.watttime.org/v2/data",
    headers=headers,
    params={'ba': 'CAISO_NORTH'}
)
```

### 定价
- **免费**: 基础功能
- **Pro**: 联系定价

---

## 7. Carbon Interface API ⭐⭐⭐
**网址**: https://www.carboninterface.com/

### 特点
- **简单易用**: RESTful API
- **多种计算**: 电力、运输、航运等
- **详细文档**: 开发友好

### API使用
```python
import requests

headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

estimate = {
    "type": "electricity",
    "electricity_unit": "kwh",
    "electricity_value": 100,
    "country": "us",
    "state": "ca"
}

response = requests.post(
    "https://www.carboninterface.com/api/v1/estimates",
    json=estimate,
    headers=headers
)
```

### 定价
- **免费**: 200次/月
- **付费**: $29起/月

---

## 8. Google Cloud Carbon Footprint ⭐⭐⭐
**网址**: https://cloud.google.com/carbon-footprint

### 特点
- **GCP集成**: 自动追踪GCP使用
- **历史数据**: 可查看趋势
- **免费**: GCP用户免费

### 使用方法
```python
# 通过GCP Console API
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{project_id}"

# 查询碳排放指标
results = client.list_time_series(
    request={
        "name": project_name,
        "filter": 'metric.type="carbon.googleapis.com/footprint"'
    }
)
```

---

## 9. Azure Sustainability Calculator ⭐⭐⭐
**网址**: https://appsource.microsoft.com/en-us/product/power-bi/coi-sustainability.sustainability-calculator

### 特点
- **Azure集成**: 自动计算Azure使用
- **Power BI报表**: 可视化
- **企业级**: 适合大规模使用

---

## 10. Hugging Face Hub（75个真实数据）⭐⭐⭐⭐
**网址**: https://huggingface.co/

### 特点
- **已收集75个**: 我们已经收集了75个有真实碳数据的模型
- **包含43个CodeCarbon**: 其中43个是专业CodeCarbon工具测量的
- **免费开放**: 无需API key
- **现成可用**: 数据已清洗整理

### 数据结构
```python
# 已收集的数据
import pandas as pd

# 75个模型的完整碳数据
all_data = pd.read_csv("data/raw/huggingface_real_carbon_data.csv")

# 43个高质量CodeCarbon数据
high_quality = pd.read_csv("data/raw/high_quality_carbon_data.csv")

# 使用示例
def find_similar_model(target_model_name):
    # 在75个真实数据中找最相似的
    similar = all_data[all_data['model_name'].str.contains(target_model_name, case=False)]
    return similar['co2_kg'].iloc[0] if len(similar) > 0 else None
```

### 实际覆盖的模型
- BERT系列: 15个变种
- GPT/生成模型: 8个
- 句子嵌入: 12个
- 多语言模型: 10个
- 其他各类模型: 30个

---

## 🎯 推荐组合使用

### 开发阶段
1. **CodeCarbon**: 实时测量你的训练
2. **ML CO2 Calculator**: 预先估算

### 生产环境
1. **Climatiq API**: 专业可靠
2. **ElectricityMap**: 实时碳强度
3. **云服务商工具**: GCP/Azure/AWS原生支持

### 研究用途
1. **Green Algorithms**: 学术标准
2. **WattTime**: 优化训练时间

---

## 💡 实用建议

### 免费方案
```python
# 组合使用免费额度
apis = {
    "climatiq": 250,      # 次/月
    "carbon_interface": 200,  # 次/月
    "electricitymap": 0,     # 需付费
    "watttime": "unlimited"   # 基础功能
}
# 总计: 450次/月免费API调用
```

### 最佳实践
1. **缓存结果**: 相同配置不重复调用
2. **批量查询**: 减少API调用次数
3. **降级策略**: API失败时用本地估算

### 集成示例
```python
class CarbonAPIManager:
    def __init__(self):
        self.apis = {
            "primary": ClimatiqAPI(),
            "backup": CarbonInterfaceAPI(),
            "fallback": LocalMLCO2Calculator()
        }

    def estimate(self, params):
        for api_name, api in self.apis.items():
            try:
                return api.calculate(params)
            except:
                continue
        return self.simple_rule_estimate(params)
```

---

## 📊 API对比表

| API | 免费额度 | 精度 | AI专用 | 实时数据 | 推荐指数 |
|-----|---------|------|--------|----------|----------|
| CodeCarbon | 无限 | 最高 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| ML CO2 | 无限 | 高 | ✅ | ❌ | ⭐⭐⭐⭐ |
| Climatiq | 250/月 | 高 | ❌ | ✅ | ⭐⭐⭐⭐ |
| Green Algorithms | 无限 | 中 | ✅ | ❌ | ⭐⭐⭐⭐ |
| ElectricityMap | 0 | 高 | ❌ | ✅ | ⭐⭐⭐ |
| Carbon Interface | 200/月 | 中 | ❌ | ❌ | ⭐⭐⭐ |
| WattTime | 部分免费 | 高 | ❌ | ✅ | ⭐⭐⭐ |
| Cloud Native | 无限* | 高 | ✅ | ✅ | ⭐⭐⭐ |

*仅限该云服务用户

---

## 🚀 快速开始

### 最简单方案
```bash
# 安装CodeCarbon
pip install codecarbon

# 在训练脚本中添加3行代码
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()
# ... 你的训练代码 ...
emissions = tracker.stop()
print(f"This training produced {emissions} kg CO2")
```

### API聚合方案
```bash
# 克隆我们的项目
git clone <your-repo>

# 配置API keys
export CLIMATIQ_API_KEY="your-key"
export CARBON_INTERFACE_KEY="your-key"

# 运行估算
python src/carbon_estimator.py --model gpt-3 --task training
```

这些API和工具的组合使用，可以覆盖从开发到生产的全部场景！