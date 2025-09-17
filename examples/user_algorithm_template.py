#!/usr/bin/env python3
"""
用户算法标准模板

按照统一范式编写的算法模板，包含所有必需的字段和函数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_miner.core.factor_storage import get_global_storage

# ==================== 算法元信息（必须） ====================
ALGORITHM_INFO = {
    # 必填字段
    'id': 'template_algorithm',                    # 算法唯一标识
    'name': '模板算法',                            # 算法显示名称
    'description': '用户算法标准模板，展示所有必需的字段和函数',  # 算法描述
    'category': 'template',                        # 算法分类
    'version': '1.0.0',                           # 版本号
    'author': 'FactorMiner Team',                 # 作者
    
    # 可选字段
    'parameters': {                                # 算法参数定义
        'window_size': {
            'type': 'int',
            'default': 20,
            'description': '时间窗口大小',
            'min': 5,
            'max': 100,
            'required': True
        },
        'threshold': {
            'type': 'float',
            'default': 0.5,
            'description': '阈值参数',
            'min': 0.0,
            'max': 1.0,
            'required': False
        },
        'method': {
            'type': 'str',
            'default': 'sma',
            'description': '计算方法',
            'choices': ['sma', 'ema', 'wma'],
            'required': True
        }
    },
    'requirements': ['pandas', 'numpy'],          # 依赖包
    'tags': ['template', 'example'],              # 标签
    'created_at': '2024-01-01',                   # 创建时间
    'updated_at': '2024-01-01'                    # 更新时间
}

# ==================== 核心函数（必须） ====================

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
    print("🚀 开始模板算法挖掘...")
    
    # 1. 数据验证
    if not validate_data(data):
        raise ValueError("数据验证失败")
    
    print(f"✅ 数据验证通过，共 {len(data)} 条记录")
    
    # 2. 获取参数
    window_size = kwargs.get('window_size', ALGORITHM_INFO['parameters']['window_size']['default'])
    threshold = kwargs.get('threshold', ALGORITHM_INFO['parameters']['threshold']['default'])
    method = kwargs.get('method', ALGORITHM_INFO['parameters']['method']['default'])
    
    print(f"📊 使用参数: window_size={window_size}, threshold={threshold}, method={method}")
    
    # 3. 特征工程
    features = create_features(data, window_size)
    print(f"✅ 特征工程完成，特征数量: {features.shape[1]}")
    
    # 4. 模型训练（如果需要）
    model = train_model(features, data, method)
    print("✅ 模型训练完成")
    
    # 5. 生成因子
    factors = {}
    factors['template_factor_1'] = generate_factor_1(features, model, threshold)
    factors['template_factor_2'] = generate_factor_2(features, model, threshold)
    
    print(f"✅ 因子生成完成，共生成 {len(factors)} 个因子")
    
    return factors

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
    print(f"🔄 计算因子: {factor_name}")
    
    # 1. 数据验证
    if not validate_data(data):
        raise ValueError("数据验证失败")
    
    # 2. 获取参数
    window_size = kwargs.get('window_size', ALGORITHM_INFO['parameters']['window_size']['default'])
    threshold = kwargs.get('threshold', ALGORITHM_INFO['parameters']['threshold']['default'])
    method = kwargs.get('method', ALGORITHM_INFO['parameters']['method']['default'])
    
    # 3. 加载预训练模型（如果需要）
    model = load_model(factor_name)
    
    # 4. 特征工程
    features = create_features(data, window_size)
    
    # 5. 计算因子
    if factor_name == 'template_factor_1':
        return calculate_factor_1(features, model, threshold)
    elif factor_name == 'template_factor_2':
        return calculate_factor_2(features, model, threshold)
    else:
        raise ValueError(f"未知因子名称: {factor_name}")

# ==================== 辅助函数（可选但推荐） ====================

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
        print("❌ 缺少必要的列")
        return False
    
    # 检查数据长度
    if len(data) < 100:  # 最少需要100个数据点
        print("❌ 数据长度不足")
        return False
    
    # 检查数据质量
    if data[required_columns].isnull().all().any():
        print("❌ 数据包含全空列")
        return False
    
    # 检查价格数据合理性
    if (data['high'] < data['low']).any():
        print("❌ 价格数据不合理（high < low）")
        return False
    
    return True

def get_factor_info(factor_name: str) -> Dict:
    """
    获取因子信息
    
    Args:
        factor_name: 因子名称
        
    Returns:
        Dict: 因子信息
    """
    factor_info = {
        'template_factor_1': {
            'name': '模板因子1',
            'description': '基于移动平均的动量因子',
            'type': 'technical',
            'parameters': {
                'window_size': 20,
                'threshold': 0.5
            }
        },
        'template_factor_2': {
            'name': '模板因子2',
            'description': '基于价格波动的反转因子',
            'type': 'statistical',
            'parameters': {
                'window_size': 20,
                'threshold': 0.5
            }
        }
    }
    
    return factor_info.get(factor_name, {})

# ==================== 算法实现函数 ====================

def create_features(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """特征工程"""
    features = pd.DataFrame(index=data.index)
    
    # 价格特征
    features['price_change'] = data['close'].pct_change()
    features['high_low_ratio'] = data['high'] / data['low']
    features['close_open_ratio'] = data['close'] / data['open']
    
    # 移动平均特征
    features['ma_short'] = data['close'].rolling(window_size // 2).mean()
    features['ma_long'] = data['close'].rolling(window_size).mean()
    features['ma_ratio'] = features['ma_short'] / features['ma_long']
    
    # 波动率特征
    features['volatility'] = data['close'].rolling(window_size).std()
    features['volatility_ratio'] = features['volatility'] / data['close'].rolling(window_size).mean()
    
    # 成交量特征
    features['volume_ma'] = data['volume'].rolling(window_size).mean()
    features['volume_ratio'] = data['volume'] / features['volume_ma']
    
    return features

def train_model(features: pd.DataFrame, data: pd.DataFrame, method: str) -> Dict:
    """模型训练"""
    # 这里是一个简单的示例，实际算法可能更复杂
    model = {
        'method': method,
        'window_size': len(features),
        'trained_at': pd.Timestamp.now()
    }
    
    # 如果需要保存模型到存储系统
    storage = get_global_storage()
    # storage.save_model("template_model", model_data, "pkl")
    
    return model

def load_model(factor_name: str) -> Dict:
    """加载预训练模型"""
    # 这里是一个简单的示例，实际应该从存储系统加载
    model = {
        'method': 'sma',
        'window_size': 20,
        'loaded_at': pd.Timestamp.now()
    }
    
    # 如果需要从存储系统加载模型
    # storage = get_global_storage()
    # model_data = storage.load_model(f"template_model_{factor_name}", "pkl")
    # model = joblib.loads(model_data)
    
    return model

def generate_factor_1(features: pd.DataFrame, model: Dict, threshold: float) -> pd.Series:
    """生成因子1"""
    # 基于移动平均的动量因子
    factor = features['ma_ratio'] - 1.0
    factor = factor.fillna(0)
    
    # 应用阈值
    factor = np.where(factor > threshold, 1, np.where(factor < -threshold, -1, 0))
    
    return pd.Series(factor, index=features.index)

def generate_factor_2(features: pd.DataFrame, model: Dict, threshold: float) -> pd.Series:
    """生成因子2"""
    # 基于价格波动的反转因子
    factor = -features['volatility_ratio']  # 负号表示反转
    factor = factor.fillna(0)
    
    # 应用阈值
    factor = np.where(factor > threshold, 1, np.where(factor < -threshold, -1, 0))
    
    return pd.Series(factor, index=features.index)

def calculate_factor_1(features: pd.DataFrame, model: Dict, threshold: float) -> pd.Series:
    """计算因子1"""
    return generate_factor_1(features, model, threshold)

def calculate_factor_2(features: pd.DataFrame, model: Dict, threshold: float) -> pd.Series:
    """计算因子2"""
    return generate_factor_2(features, model, threshold)

# ==================== 测试函数 ====================

def test_algorithm():
    """测试算法"""
    print("🧪 测试模板算法...")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    try:
        # 测试挖掘阶段
        print("\n=== 测试挖掘阶段 ===")
        factors = calculate_factors(data, window_size=20, threshold=0.3, method='sma')
        print(f"生成的因子: {list(factors.keys())}")
        
        # 测试计算阶段
        print("\n=== 测试计算阶段 ===")
        factor_1 = calculate_single_factor(data, 'template_factor_1', window_size=20)
        factor_2 = calculate_single_factor(data, 'template_factor_2', window_size=20)
        
        print(f"因子1长度: {len(factor_1)}")
        print(f"因子2长度: {len(factor_2)}")
        
        print("\n✅ 算法测试成功！")
        
    except Exception as e:
        print(f"❌ 算法测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_algorithm()
