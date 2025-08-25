import asyncio

from mcp import StdioServerParameters, stdio_client, ClientSession


async def test_mcp_server():
    """测试时间服务器"""

    # 配置服务器参数
    server_params = StdioServerParameters(
        command="python",
        args=["mining_server.py"]
    )

    print("🔗 连接到服务器...")

    # 创建客户端会话
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()

            print("✅ 连接成功！")

            # 获取可用工具列表
            tool_list = await session.list_tools()
            print(f"📋 可用工具: {[tool.name for tool in tool_list.tools]}")

            # 调用启动挖矿任务
            result = await session.call_tool("run_mining", {
                "symbol": "BTC_USDT",
                "timeframe": "1m",
                "factor_types": ["technical"],
                "start_date": "2025-07-20",
                "end_date": "2025-08-19"
            })
            print(f"⛏️ 挖矿任务启动结果: {result.content[0].text}")




if __name__ == '__main__':
    asyncio.run(test_mcp_server())
