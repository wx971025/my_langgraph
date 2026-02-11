import json
from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel, Field

from utils.decorator import time_record

repl = PythonREPL()

class PythonREPLToolSchema(BaseModel):
    code: str = Field(..., description="The python code to execute to generate your chart.")

@tool(args_schema=PythonREPLToolSchema)
@time_record
def python_repl_tool(code: str) -> str:
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str
