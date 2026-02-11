from langchain_core.tools import tool
from pydantic import BaseModel, Field

class DBAgentToolSchema(BaseModel):
    query: str = Field(..., description="The specific query or instruction for the database agent.")

@tool(name_or_callable="db_agent", args_schema=DBAgentToolSchema)
def db_agent_tool(query: str) -> str:
    """You are a database specialist. You can query, add, update, or delete sales data.
    If the user asks for something outside your scope (like data analysis, plotting, or general coding),
    explicitly state that you cannot do it and return your findings so far.
    Do NOT ask the user for clarification if the request is simply out of your scope.
    """
    pass
