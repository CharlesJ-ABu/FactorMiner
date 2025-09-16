# 用户算法目录

本目录包含所有因子挖掘算法，包括预设算法和用户自定义算法。

## 算法编写规范

### 标准接口

每个算法文件必须实现以下接口：

```python
def calculate_factors(data, **kwargs):
    """
    计算因子
    
    Args:
        data: DataFrame，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数
    
    Returns:
        Dict[str, pd.Series]: 因子名称 -> 因子值序列
    """
    # 算法实现
    return {
        'factor_name': factor_series,
        # ...
    }

# 算法元信息（可选）
ALGORITHM_INFO = {
    'name': '算法名称',
    'description': '算法描述',
    'category': '算法类别',
    'parameters': {
        'param_name': {
            'type': 'int/float/str/bool',
            'default': 默认值,
            'description': '参数描述'
        }
    }
}
```

### 命名约定

- 文件名使用下划线分隔：`category_algorithm_name.py`
- 例如：`ml_random_forest.py`, `statistical_zscore.py`

### 算法类别

- `ml_*`: 机器学习算法
- `statistical_*`: 统计因子算法  
- `advanced_*`: 高级因子算法
- `custom_*`: 用户自定义算法
