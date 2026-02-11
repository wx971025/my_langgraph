import math
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("Divisor cannot be zero")
    return a / b

@mcp.tool()
def square(a: int, b: int) -> int:
    return math.pow(a, b)

if __name__ == "__main__":
    mcp.run(transport="stdio")
