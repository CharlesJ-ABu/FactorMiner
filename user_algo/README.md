# 用户算法目录

> 🚀 **项目状态**: 当前重构因子挖掘模块中，bug 很多，需要测试的同学可以找到 9 月份之前的版本进行测试

本目录包含所有因子挖掘算法，包括预设算法和用户自定义算法。

**⚠️ 重要提醒**: 当前正在重构用户算法范式，请参考最新的开发文档。

## 🔄 新算法编写规范

### 必填接口

每个算法文件必须实现以下接口：

```python
# 1. 算法元信息 (必填)
ALGORITHM_INFO = {
    # 必填字段
    'id': 'algorithm_id',
    'name': '算法名称',
    'description': '算法描述',
    'category': '算法类别',
    'version': '1.0.0',
    'author': '作者名称',
    
    # 可选字段
    'parameters': {
        'param_name': {
            'type': 'int/float/str/bool/list',
            'default': 默认值,
            'description': '参数描述',
            'min': 最小值,
            'max': 最大值,
            'required': False
        }
    },
    'requirements': ['pandas', 'numpy'],
    'tags': ['tag1', 'tag2'],
    'created_at': '2024-01-01',
    'updated_at': '2024-01-01'
}

# 2. 因子挖掘函数 (必填)
def calculate_factors(data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """
    挖掘阶段：训练模型并生成所有因子
    系统会调用这个函数进行因子挖掘
    
    Args:
        data: OHLCV数据，包含 'open', 'high', 'low', 'close', 'volume' 列
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
    
    Returns:
        Dict[str, pd.Series]: 因子名称到因子值的映射
    """
    # 算法实现
    return {
        'factor_name': factor_series,
        # ...
    }

# 3. 单因子计算函数 (必填)
def calculate_single_factor(data: pd.DataFrame, factor_name: str, **kwargs) -> pd.Series:
    """
    计算阶段：计算单个因子
    系统会调用这个函数进行实时因子计算
    
    Args:
        data: OHLCV数据，包含 'open', 'high', 'low', 'close', 'volume' 列
        factor_name: 因子名称，如 'factor_name'
        **kwargs: 算法参数，来自ALGORITHM_INFO['parameters']
        
    Returns:
        pd.Series: 因子值序列，索引与data相同
    """
    # 算法实现
    return pd.Series(factor_values, index=data.index, name=factor_name)
```

### 推荐辅助函数

```python
# 数据验证函数 (推荐)
def validate_data(data: pd.DataFrame) -> bool:
    """数据验证"""
    # 实现数据验证逻辑
    return True

# 因子信息获取函数 (推荐)
def get_factor_info(factor_name: str) -> Dict:
    """获取因子信息"""
    # 返回因子相关信息
    return {}
```

### 命名约定

- 文件名使用下划线分隔：`category_algorithm_name.py`
- 例如：`ml_random_forest.py`, `statistical_zscore.py`

### 算法类别

- `ml_*`: 机器学习算法
- `statistical_*`: 统计因子算法  
- `advanced_*`: 高级因子算法
- `custom_*`: 用户自定义算法

---

*最后更新: 2025年9月*
