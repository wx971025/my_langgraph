import os
import json
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_config = json.load(open("tools/config.json"))

async def get_local_mcp_tools():
    try:
        client = MultiServerMCPClient(mcp_config)
        mcp_tools = await client.get_tools()
        print(f"成功加载{len(mcp_tools)}个工具: \n{[t.name for t in mcp_tools]}")
        return mcp_tools
    except Exception as e:
        print(f"加载工具失败: {e}")
        return []
