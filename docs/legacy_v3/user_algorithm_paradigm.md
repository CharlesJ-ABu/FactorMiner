# 用户算法统一范式

## 概述

为了确保因子挖掘系统的统一性和可维护性，所有用户算法必须遵循统一的范式。这个范式定义了算法文件必须包含的字段、函数和结构。

## 范式规范

### 1. 文件结构要求

每个用户算法文件必须包含以下部分：

```python
# 1. 文档字符串
"""
算法名称和描述
"""

# 2. 导入语句
import pandas as pd
import numpy as np
# ... 其他导入

# 3. 算法元信息（必须）
ALGORITHM_INFO = {
    # 必填字段
    'id': 'unique_algorithm_id',           # 算法唯一标识
    'name': '算法显示名称',                 # 算法名称
    'description': '算法详细描述',          # 算法描述
    'category': 'algorithm_category',      # 算法分类
    'version': '1.0.0',                    # 版本号
    'author': '作者名称',                   # 作者
    
    # 可选字段
    'parameters': {                         # 算法参数定义
        'param1': {
            'type': 'int',
            'default': 10,
            'description': '参数描述',
            'min': 1,
            'max': 100
        }
    },
    'requirements': ['pandas', 'numpy'],   # 依赖包
    'tags': ['ml', 'momentum'],            # 标签
    'created_at': '2024-01-01',            # 创建时间
    'updated_at': '2024-01-01'             # 更新时间
}

# 4. 核心函数（必须）
def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """挖掘阶段：训练模型并生成所有因子"""
    pass

def calculate_single_factor(data: pd.DataFrame, factor_name: str, **kwargs) -> pd.Series:
    """计算阶段：计算单个因子"""
    pass

# 5. 辅助函数（可选）
def validate_data(data: pd.DataFrame) -> bool:
    """数据验证"""
    pass

def get_factor_info(factor_name: str) -> Dict:
    """获取因子信息"""
    pass
```

### 2. 必填字段详解

#### ALGORITHM_INFO 必填字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `id` | str | 算法唯一标识，用于系统识别 | `'ml_momentum_factor'` |
| `name` | str | 算法显示名称 | `'ML动量因子'` |
| `description` | str | 算法详细描述 | `'基于机器学习的动量因子挖掘算法'` |
| `category` | str | 算法分类 | `'ml'`, `'technical'`, `'statistical'` |
| `version` | str | 版本号，遵循语义化版本 | `'1.0.0'` |
| `author` | str | 作者名称 | `'FactorMiner Team'` |

#### ALGORITHM_INFO 可选字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `parameters` | dict | 算法参数定义 | 见下方参数定义格式 |
| `requirements` | list | 依赖包列表 | `['pandas', 'numpy', 'sklearn']` |
| `tags` | list | 标签列表 | `['ml', 'momentum', 'regression']` |
| `created_at` | str | 创建时间 | `'2024-01-01'` |
| `updated_at` | str | 更新时间 | `'2024-01-01'` |

### 3. 参数定义格式

```python
'parameters': {
    'param_name': {
        'type': 'int|float|str|bool|list',  # 参数类型
        'default': default_value,           # 默认值
        'description': '参数描述',          # 参数描述
        'min': min_value,                  # 最小值（数值类型）
        'max': max_value,                  # 最大值（数值类型）
        'choices': [choice1, choice2],     # 可选值（列表类型）
        'required': True|False             # 是否必需
    }
}
```

### 4. 核心函数规范

#### calculate_factors 函数

**用途**: 挖掘阶段，训练模型并生成所有因子
**调用时机**: 用户启动因子挖掘时
**返回值**: `Dict[str, pd.Series]` - 因子名称到因子值的映射

```python
def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """
    挖掘阶段：训练模型并生成所有因子
    系统会调用这个函数进行因子挖掘
    
    Args:
        data: OHLCV数据，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
        
    Returns:
        Dict[str, pd.Series]: 因子名称到因子值的映射
        例如: {
            'factor_1': pd.Series(...),
            'factor_2': pd.Series(...)
        }
    """
    # 1. 数据验证
    if not validate_data(data):
        raise ValueError("数据验证失败")
    
    # 2. 特征工程
    features = create_features(data)
    
    # 3. 模型训练（如果需要）
    model = train_model(features, data)
    
    # 4. 生成因子
    factors = {}
    factors['factor_1'] = generate_factor_1(features, model)
    factors['factor_2'] = generate_factor_2(features, model)
    
    return factors
```

#### calculate_single_factor 函数

**用途**: 计算阶段，计算单个因子
**调用时机**: 用户请求计算特定因子时
**返回值**: `pd.Series` - 因子值序列

```python
def calculate_single_factor(data: pd.DataFrame, factor_name: str, **kwargs) -> pd.Series:
    """
    计算阶段：计算单个因子
    系统会调用这个函数进行实时因子计算
    
    Args:
        data: OHLCV数据，包含 'open', 'high', 'low', 'close', 'volume' 列
        factor_name: 因子名称，来自calculate_factors返回的键
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
        
    Returns:
        pd.Series: 因子值序列，索引与data相同
    """
    # 1. 数据验证
    if not validate_data(data):
        raise ValueError("数据验证失败")
    
    # 2. 加载预训练模型（如果需要）
    model = load_model(factor_name)
    
    # 3. 特征工程
    features = create_features(data)
    
    # 4. 计算因子
    if factor_name == 'factor_1':
        return calculate_factor_1(features, model)
    elif factor_name == 'factor_2':
        return calculate_factor_2(features, model)
    else:
        raise ValueError(f"未知因子名称: {factor_name}")
```

### 5. 辅助函数规范

#### validate_data 函数

```python
def validate_data(data: pd.DataFrame) -> bool:
    """
    数据验证
    
    Args:
        data: 输入数据
        
    Returns:
        bool: 验证是否通过
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 检查必要列
    if not all(col in data.columns for col in required_columns):
        return False
    
    # 检查数据长度
    if len(data) < 100:  # 最少需要100个数据点
        return False
    
    # 检查数据质量
    if data[required_columns].isnull().all().any():
        return False
    
    return True
```

#### get_factor_info 函数

```python
def get_factor_info(factor_name: str) -> Dict:
    """
    获取因子信息
    
    Args:
        factor_name: 因子名称
        
    Returns:
        Dict: 因子信息
    """
    factor_info = {
        'factor_1': {
            'name': '因子1',
            'description': '因子1的描述',
            'type': 'ml',
            'parameters': {}
        },
        'factor_2': {
            'name': '因子2', 
            'description': '因子2的描述',
            'type': 'technical',
            'parameters': {}
        }
    }
    
    return factor_info.get(factor_name, {})
```

### 6. 算法分类标准

| 分类 | 说明 | 示例 |
|------|------|------|
| `ml` | 机器学习算法 | 随机森林、神经网络、SVM |
| `technical` | 技术指标算法 | 移动平均、RSI、MACD |
| `statistical` | 统计算法 | 相关性、协整、主成分分析 |
| `advanced` | 高级算法 | 深度学习、强化学习 |

### 7. 错误处理规范

```python
def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    try:
        # 算法逻辑
        pass
    except Exception as e:
        # 记录错误
        print(f"❌ 算法执行失败: {e}")
        # 返回空结果而不是抛出异常
        return {}
```

### 8. 日志规范

```python
def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    print("🚀 开始算法执行...")
    print(f"✅ 数据加载完成，共 {len(data)} 条记录")
    print(f"🔄 开始特征工程...")
    print(f"✅ 特征工程完成，特征数量: {features.shape[1]}")
    print(f"🔄 开始模型训练...")
    print(f"✅ 模型训练完成")
    print(f"🔄 开始因子生成...")
    print(f"✅ 因子生成完成，共生成 {len(factors)} 个因子")
```

## 完整示例

参考 `examples/user_algorithm_template.py` 查看完整的算法模板。

## 验证工具

使用 `scripts/validate_algorithm.py` 验证算法是否符合范式要求。
