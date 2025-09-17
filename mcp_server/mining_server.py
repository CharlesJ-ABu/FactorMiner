
import asyncio
import json

import mcp
from mcp.server import Server, InitializationOptions, NotificationOptions
from mcp.types import TextContent, Tool
import sys
from pathlib import Path

# 获取当前文件的目录
current_dir = Path(__file__).parent

project_root = current_dir.parent
# 将项目根目录添加到 Python 模块搜索路径
sys.path.append(str(project_root))

# 现在再导入新的API
from factor_miner.api.factor_mining_api import FactorMiningAPI

# 创建MCP服务器实例
server = Server("mining-server")
mining_api = FactorMiningAPI()

# 定义一个获取当前时间的工具
@server.list_tools()
async def list_tools():
    """列出所有可用的工具"""
    return [
        Tool(
            name="run_mining",
            description="启动挖掘任务",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易对符号"},
                    "timeframe": {"type": "string", "description": "时间周期"},
                    "factor_types": {"type": "array", "description": "因子类型列表"},
                    "start_date": {"type": "string", "description": "开始日期"},
                    "end_date": {"type": "string", "description": "结束日期"}
                },
                "required": ["symbol", "timeframe", "factor_types", "start_date", "end_date"]
            }
        ),
        Tool(
            name="get_mining_progress",
            description="获取挖掘任务进度",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务ID"}
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="get_mining_result",
            description="获取挖掘任务结果",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务ID"}
                },
                "required": ["task_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """执行工具调用"""
    if name == "run_mining":
        symbol = arguments.get("symbol")
        timeframe = arguments.get("timeframe")
        factor_types = arguments.get("factor_types")
        start_date = arguments.get("start_date")
        end_date = arguments.get("end_date")
        # 调用新的API方法
        mining_result = mining_api.run_complete_mining(
            symbol=symbol,
            timeframe=timeframe,
            factor_types=factor_types,
            start_date=start_date,
            end_date=end_date
        )

        return [TextContent(
            type="text",
            text=f"挖掘任务已启动，任务ID: {mining_result}"
        )]

    raise ValueError(f"未知的工具: {name}")

# 启动服务器的函数
async def main():
    """启动MCP服务器"""
    print("🚀 启动服务器...")
    print("服务器正在监听连接...")

    # 这里我们使用标准输入输出作为传输方式
    # 在实际应用中，你可能会使用WebSocket或HTTP
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mining-server",
                server_version="1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
