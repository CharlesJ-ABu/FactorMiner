# tests/test_mining_service.py

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import uuid
from webui.service.mining_service import MiningService


class TestMiningService(unittest.TestCase):
    def setUp(self):
        """在每个测试方法前初始化 MiningService 实例"""
        self.mining_service = MiningService()

    def test_start_mining_with_valid_params(self):
        """测试 start_mining 方法（使用有效参数）"""
        valid_params = {
            'symbols': ['BTC_USDT'],
            'timeframes': ['1m'],
            'factor_types': ['technical'],
            'start_date': '2025-08-01',
            'end_date': '2025-08-20'
        }

        # 调用 start_mining
        result = self.mining_service.start_mining(valid_params)

        # 断言返回结果结构
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('session_id', result)
        self.assertIn('message', result)
        self.assertIn('estimated_time', result)
        self.assertIn('system_info', result)

        # success 应为 True 表示启动成功
        self.assertTrue(result['success'], msg="挖掘启动应成功")

        # session_id 应为非空（注意：当前代码有 bug，返回的是 self.mining_sessions，应该返回 session_id 字符串）
        session_id = result['session_id']
        # ⚠️ 注意：您的 start_mining() 方法中这一行有误：
        # return {'success': True, 'session_id': self.mining_sessions, ...}
        # 应该是：
        # return {'success': True, 'session_id': session_id, ...}
        # 所以此处根据您的实际返回值调整断言
        # 暂时假设您已修复，返回的是 session_id 字符串
        if isinstance(session_id, dict):  # 当前有 bug，返回的是 self.mining_sessions（字典）
            self.fail("start_mining() 返回的 session_id 错误，应为字符串，实际返回了整个 mining_sessions 字典。请检查代码。")
        else:
            self.assertIsInstance(session_id, str, msg="session_id 应为字符串")

        # 检查 mining_sessions 中是否存在该 session
        self.assertIn(session_id, self.mining_service.mining_sessions, msg="启动的 session 应存在于 mining_sessions 中")

        # 检查状态是否为 pending 或 running（取决于实现）
        session = self.mining_service.mining_sessions[session_id]
        self.assertIn(session['status'], ['pending', 'running', 'completed'])

    def test_start_mining_with_missing_params(self):
        """测试 start_mining 方法（缺少必要参数）"""
        invalid_params = {
            'symbols': ['BTCUSDT'],
            'timeframes': ['1h'],
            # 缺少 factor_types, start_date, end_date
        }

        result = self.mining_service.start_mining(invalid_params)

        self.assertIsInstance(result, dict)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertIn('缺少必要参数', result['error'])

    @patch.object(MiningService, 'get_mining_api')
    @patch.object(MiningService, 'get_ml_factor_builder')
    def test_get_mining_status_for_valid_session(self, mock_ml_builder, mock_mining_api):
        """测试 get_mining_status 方法（使用有效的 session_id）"""
        # 模拟一个已有的 session
        session_id = str(uuid.uuid4())
        mock_session = {
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'config': {},
            'progress': {'data_loading': 50},
            'progress_info': {},
            'current_step': 'data_loading',
            'messages': [],
            'system_info': {}
        }
        self.mining_service.mining_sessions[session_id] = mock_session

        # 调用 get_mining_status
        status = self.mining_service.get_mining_status(session_id)

        self.assertIsInstance(status, dict)
        self.assertTrue(status['success'])
        self.assertEqual(status['session_id'], session_id)
        self.assertEqual(status['status'], 'running')
        self.assertIn('progress', status)
        self.assertIn('current_step', status)

    def test_get_mining_status_for_invalid_session(self):
        """测试 get_mining_status 方法（使用无效的 session_id）"""
        invalid_session_id = 'non_existent_session_123'

        status = self.mining_service.get_mining_status(invalid_session_id)

        self.assertIsInstance(status, dict)
        self.assertFalse(status['success'])
        self.assertIn('error', status)
        self.assertEqual(status['error'], '挖掘会话不存在')

    def test_estimate_step_time_for_data_loading(self):
        """测试 estimate_step_time 方法（data_loading 步骤）"""
        step_name = 'data_loading'
        config = {}
        data_info = {'data_size_mb': 100}  # 100MB

        time_est, steps = self.mining_service.estimate_step_time(step_name, config, data_info)

        self.assertGreater(time_est, 5)  # 至少基础时间 5s + 数据大小相关时间
        self.assertGreaterEqual(steps, 1)
        self.assertLessEqual(steps, 10)  # 不超过 max_progress_steps

    def test_estimate_step_time_for_unknown_step(self):
        """测试 estimate_step_time 方法（未知步骤）"""
        step_name = 'unknown_step'
        config = {}
        time_est, steps = self.mining_service.estimate_step_time(step_name, config)

        self.assertEqual(time_est, 10)  # 默认时间
        self.assertEqual(steps, 10)     # 默认步数


if __name__ == '__main__':
    unittest.main()
