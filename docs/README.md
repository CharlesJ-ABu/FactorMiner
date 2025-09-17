# 📚 FactorMiner 项目文档

> 🚀 **项目状态**: 当前重构因子挖掘模块中，bug 很多，需要测试的同学可以找到 9 月份之前的版本进行测试

## 🎯 概述

FactorMiner 是一个专业的量化因子挖掘、评估和优化平台。本文档目录包含了项目的核心文档和指南。

**⚠️ 重要提醒**: 当前项目正在进行大规模重构，包括：
- 因子存储架构重新设计
- 用户算法范式统一
- 前端后端接口优化
- 代码结构简化

建议在生产环境中使用稳定版本。

## 📁 文档结构

### 🚀 核心文档

#### [API文档](api.md)
- **内容**: 完整的API接口文档
- **用途**: 开发者参考，程序化调用指南
- **状态**: ⚠️ 重构中，部分接口可能已变更

#### [因子存储架构指南](factor_storage_guide.md)
- **内容**: V3因子存储架构详解
- **用途**: 理解因子存储和管理方式
- **状态**: ⚠️ 重构中，架构正在重新设计

### 🔄 重构相关文档

#### [新架构指南](new_architecture_guide.md)
- **内容**: 重构后的新架构设计
- **用途**: 理解新的系统架构和设计理念
- **状态**: ✅ 最新，反映重构方向

#### [用户算法范式](user_algorithm_paradigm.md)
- **内容**: 统一的用户算法开发规范
- **用途**: 用户自定义算法开发指南
- **状态**: ✅ 最新，重构核心内容

#### [存储API指南](storage_api_guide.md)
- **内容**: 简化的因子存储API使用指南
- **用途**: 开发者使用新的存储接口
- **状态**: ✅ 最新，重构核心内容

#### [因子库结构指南](factorlib_structure_guide.md)
- **内容**: 新的factorlib目录结构说明
- **用途**: 理解因子库的组织方式
- **状态**: ✅ 最新，重构核心内容



### 🔧 实用指南



#### [VPN设置指南](vpn_setup.md)
- **内容**: 网络配置和VPN设置
- **用途**: 解决网络访问问题
- **状态**: ✅ 实用指南，保持更新

## 🗑️ 已清理的过时文档

以下文档已被删除，因为它们描述的功能已完成或架构已过时：

- `future_function_fix_report.md` - 功能修复已完成
- `win_rate_calculation_fix.md` - 胜率计算修复已完成
- `factorlib_migration_summary.md` - 迁移已完成
- `factor_name_cleanup_report.md` - 清理已完成
- `factor_cleanup_report.md` - 清理已完成
- `modal_fix.md` - 模态框修复已完成
- `factor_storage_architecture.md` - 已被新架构替代
- `new_factor_storage_architecture.md` - 部分内容过时
- `user_guide.md` - 空文件
- `factor_details_ui_improvements.md` - 空文件
- `factor_library_design.md` - UI已重新设计
- `factor_library_guide.md` - 已过时
- `factor_library_inline_display_guide.md` - 已过时
- `factor_library_layout_improvements.md` - 已过时
- `ui_styles.md` - 多风格功能未实现

## 🎯 文档维护原则

1. **保持最新**: 所有文档必须反映当前项目状态
2. **实用优先**: 重点维护用户和开发者需要的文档
3. **定期清理**: 及时删除过时和不再相关的文档
4. **结构清晰**: 文档组织要便于查找和使用

## 📝 如何贡献文档

1. **更新现有文档**: 当功能发生变化时，及时更新相关文档
2. **创建新文档**: 为新功能或重要概念创建文档
3. **删除过时文档**: 及时清理不再相关的文档
4. **保持一致性**: 确保文档风格和格式统一

---

*最后更新: 2025年9月*
