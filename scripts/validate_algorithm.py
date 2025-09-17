#!/usr/bin/env python3
"""
用户算法验证工具

验证用户算法是否符合统一范式要求
"""

import sys
import os
import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AlgorithmValidator:
    """算法验证器"""
    
    def __init__(self):
        self.required_fields = [
            'id', 'name', 'description', 'category', 'version', 'author'
        ]
        
        self.required_functions = [
            'calculate_factors',
            'calculate_single_factor'
        ]
        
        self.optional_functions = [
            'validate_data',
            'get_factor_info'
        ]
        
        self.errors = []
        self.warnings = []
    
    def validate_file(self, file_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        验证算法文件
        
        Args:
            file_path: 算法文件路径
            
        Returns:
            Tuple[bool, List[str], List[str]]: (是否通过, 错误列表, 警告列表)
        """
        self.errors = []
        self.warnings = []
        
        try:
            # 1. 解析文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 2. 检查ALGORITHM_INFO
            self._validate_algorithm_info(tree)
            
            # 3. 检查必需函数
            self._validate_required_functions(tree)
            
            # 4. 检查函数签名
            self._validate_function_signatures(tree)
            
            # 5. 检查导入语句
            self._validate_imports(tree)
            
            # 6. 检查代码质量
            self._validate_code_quality(tree)
            
            return len(self.errors) == 0, self.errors, self.warnings
            
        except Exception as e:
            self.errors.append(f"文件解析失败: {e}")
            return False, self.errors, self.warnings
    
    def _validate_algorithm_info(self, tree: ast.AST):
        """验证ALGORITHM_INFO"""
        algorithm_info = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'ALGORITHM_INFO':
                        if isinstance(node.value, ast.Dict):
                            algorithm_info = self._extract_dict(node.value)
                        break
        
        if algorithm_info is None:
            self.errors.append("缺少ALGORITHM_INFO定义")
            return
        
        # 检查必填字段
        for field in self.required_fields:
            if field not in algorithm_info:
                self.errors.append(f"ALGORITHM_INFO缺少必填字段: {field}")
        
        # 检查字段类型
        if 'id' in algorithm_info and not isinstance(algorithm_info['id'], str):
            self.errors.append("ALGORITHM_INFO['id']必须是字符串")
        
        if 'version' in algorithm_info and not isinstance(algorithm_info['version'], str):
            self.errors.append("ALGORITHM_INFO['version']必须是字符串")
        
        if 'category' in algorithm_info:
            valid_categories = ['ml', 'technical', 'statistical', 'advanced', 'template']
            if algorithm_info['category'] not in valid_categories:
                self.warnings.append(f"ALGORITHM_INFO['category']建议使用标准分类: {valid_categories}")
        
        # 检查参数定义
        if 'parameters' in algorithm_info:
            self._validate_parameters(algorithm_info['parameters'])
    
    def _validate_parameters(self, parameters: Dict):
        """验证参数定义"""
        for param_name, param_def in parameters.items():
            if not isinstance(param_def, dict):
                self.errors.append(f"参数'{param_name}'定义必须是字典")
                continue
            
            # 检查必需字段
            required_param_fields = ['type', 'default', 'description']
            for field in required_param_fields:
                if field not in param_def:
                    self.errors.append(f"参数'{param_name}'缺少字段: {field}")
            
            # 检查类型字段
            if 'type' in param_def:
                valid_types = ['int', 'float', 'str', 'bool', 'list']
                if param_def['type'] not in valid_types:
                    self.errors.append(f"参数'{param_name}'类型无效: {param_def['type']}")
    
    def _validate_required_functions(self, tree: ast.AST):
        """验证必需函数"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        # 检查必需函数
        for func_name in self.required_functions:
            if func_name not in functions:
                self.errors.append(f"缺少必需函数: {func_name}")
        
        # 检查可选函数
        for func_name in self.optional_functions:
            if func_name not in functions:
                self.warnings.append(f"建议添加函数: {func_name}")
    
    def _validate_function_signatures(self, tree: ast.AST):
        """验证函数签名"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == 'calculate_factors':
                    self._validate_calculate_factors_signature(node)
                elif node.name == 'calculate_single_factor':
                    self._validate_calculate_single_factor_signature(node)
    
    def _validate_calculate_factors_signature(self, node: ast.FunctionDef):
        """验证calculate_factors函数签名"""
        args = [arg.arg for arg in node.args.args]
        
        if 'data' not in args:
            self.errors.append("calculate_factors函数必须包含'data'参数")
        
        if not node.args.kwarg:
            self.warnings.append("calculate_factors函数建议包含'**kwargs'参数")
        
        # 检查返回类型注解
        if node.returns is None:
            self.warnings.append("calculate_factors函数建议添加返回类型注解")
    
    def _validate_calculate_single_factor_signature(self, node: ast.FunctionDef):
        """验证calculate_single_factor函数签名"""
        args = [arg.arg for arg in node.args.args]
        
        if 'data' not in args:
            self.errors.append("calculate_single_factor函数必须包含'data'参数")
        
        if 'factor_name' not in args:
            self.errors.append("calculate_single_factor函数必须包含'factor_name'参数")
        
        if not node.args.kwarg:
            self.warnings.append("calculate_single_factor函数建议包含'**kwargs'参数")
    
    def _validate_imports(self, tree: ast.AST):
        """验证导入语句"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # 检查必需导入
        required_imports = ['pandas', 'numpy']
        for imp in required_imports:
            if not any(imp in import_name for import_name in imports):
                self.warnings.append(f"建议导入: {imp}")
    
    def _validate_code_quality(self, tree: ast.AST):
        """验证代码质量"""
        # 检查是否有适当的错误处理
        has_try_except = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                has_try_except = True
                break
        
        if not has_try_except:
            self.warnings.append("建议添加错误处理（try-except）")
        
        # 检查是否有适当的日志输出
        has_print = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    has_print = True
                    break
        
        if not has_print:
            self.warnings.append("建议添加日志输出（print语句）")
    
    def _extract_dict(self, node: ast.Dict) -> Dict:
        """从AST节点提取字典"""
        result = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant):
                key_name = key.value
            elif isinstance(key, ast.Str):  # Python < 3.8
                key_name = key.s
            else:
                continue
            
            if isinstance(value, ast.Constant):
                result[key_name] = value.value
            elif isinstance(value, ast.Str):  # Python < 3.8
                result[key_name] = value.s
            elif isinstance(value, ast.Dict):
                result[key_name] = self._extract_dict(value)
            elif isinstance(value, ast.List):
                result[key_name] = self._extract_list(value)
            else:
                result[key_name] = str(value)
        
        return result
    
    def _extract_list(self, node: ast.List) -> List:
        """从AST节点提取列表"""
        result = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                result.append(elt.value)
            elif isinstance(elt, ast.Str):  # Python < 3.8
                result.append(elt.s)
            else:
                result.append(str(elt))
        return result

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python validate_algorithm.py <algorithm_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)
    
    validator = AlgorithmValidator()
    is_valid, errors, warnings = validator.validate_file(file_path)
    
    print(f"验证文件: {file_path}")
    print("=" * 50)
    
    if errors:
        print("❌ 错误:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("⚠️  警告:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if is_valid:
        print("✅ 算法验证通过！")
    else:
        print("❌ 算法验证失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
