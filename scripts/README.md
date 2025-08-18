# 🛠️ FactorMiner 实用脚本

本文件夹包含了 FactorMiner 平台的各种实用脚本，用于维护、分析和优化平台功能。

## 🎯 脚本文件说明

### 维护和修复脚本

#### 1. **repair_empty_evaluations.py** - 修复空评估结果
- **用途**: 修复因子库中空的评估结果文件
- **功能**: 
  - 检测空的评估结果JSON文件
  - 重新运行评估计算
  - 保存正确的评估结果
- **适用场景**: 当因子评估结果丢失或为空时
- **运行方式**: `python repair_empty_evaluations.py`

#### 2. **clean_factor_names.py** - 清理因子名称
- **用途**: 清理因子库中不合理的币种后缀
- **功能**:
  - 移除因子名称末尾的币种后缀
  - 标准化symbol字段
  - 备份原文件并清理
- **适用场景**: 因子名称规范化
- **运行方式**: `python clean_factor_names.py`

### 分析和监控脚本

#### 3. **factorlib_health_check.py** - 因子库健康检查
- **用途**: 检查因子库中所有因子的运行状态
- **功能**:
  - 测试每个因子的计算能力
  - 检查数据完整性和性能
  - 生成健康检查报告
- **适用场景**: 定期维护和问题诊断
- **运行方式**: `python factorlib_health_check.py`
- **输出**: 保存到 `factorlib/exports/factorlib_health_report.json`

#### 4. **factor_analysis_tool.py** - 因子分析工具
- **用途**: 分析因子库中的因子质量和性能
- **功能**:
  - 因子质量分析（类型分布、分类统计）
  - 因子性能分析（IC值、胜率、稳定性）
  - 生成因子库报告
- **适用场景**: 因子库质量评估和优化
- **运行方式**: `python factor_analysis_tool.py`
- **输出**: 
  - 性能分析结果保存为CSV
  - 因子库报告保存为JSON

### 因子注册脚本

#### 5. **register_hazel_factors.py** - Hazel因子注册脚本
- **用途**: 批量注册Hazel技术因子到因子库
- **功能**:
  - 自动生成因子定义文件
  - 自动生成因子函数文件
  - 批量注册到factorlib
- **适用场景**: 批量添加技术因子
- **运行方式**: `python register_hazel_factors.py`

## 🚀 使用方法

### 环境准备
```bash
# 确保在项目根目录下
cd /path/to/FactorMiner

# 检查Python路径
python -c "import sys; print(sys.path)"
```

### 运行脚本
```bash
# 进入scripts目录
cd scripts

# 运行特定脚本
python repair_empty_evaluations.py
python factorlib_health_check.py
python factor_analysis_tool.py
python register_hazel_factors.py
```

### 批量运行
```bash
# 运行所有维护脚本
for script in repair_empty_evaluations.py clean_factor_names.py; do
    echo "运行 $script..."
    python "$script"
done
```

## 📊 脚本输出

### 控制台输出
- ✅ 成功操作
- ❌ 失败操作
- 📊 统计信息
- 🔍 检查结果

### 文件输出
- **健康检查**: `factorlib/exports/factorlib_health_report.json`
- **性能分析**: `factorlib/exports/factor_performance_analysis_YYYYMMDD_HHMMSS.csv`
- **因子报告**: `factorlib/exports/factor_library_report_YYYYMMDD_HHMMSS.json`
- **备份文件**: 原文件名加上时间戳后缀

## ⚠️ 注意事项

### 安全提醒
1. **备份重要数据**: 某些脚本会修改原文件，建议先备份
2. **权限检查**: 确保有写入factorlib目录的权限
3. **数据完整性**: 运行前检查数据文件的完整性

### 运行建议
1. **定期运行**: 建议定期运行健康检查和清理脚本
2. **问题诊断**: 遇到问题时优先运行健康检查脚本
3. **结果分析**: 仔细分析脚本输出的结果和警告

## 🔧 自定义修改

### 修改参数
- 在脚本中修改文件路径、时间范围等参数
- 调整检查的严格程度和阈值
- 自定义输出格式和内容

### 添加功能
- 集成新的检查项目
- 添加自定义的分析指标
- 实现自动化的维护流程

## 🆘 常见问题

### Q: 脚本运行时出现权限错误？
A: 检查文件权限，确保有读写权限

### Q: 健康检查报告为空？
A: 检查是否有可用的数据文件，确保数据路径正确

### Q: 因子分析工具运行缓慢？
A: 对于大型因子库，分析可能需要较长时间，请耐心等待

### Q: 脚本输出文件在哪里？
A: 默认保存在 `factorlib/exports/` 目录下

## 📞 技术支持

如果遇到问题，请：
1. 检查错误信息和日志
2. 确认项目配置正确
3. 参考项目文档
4. 提交Issue到项目仓库

---

*最后更新: 2025年8月*
