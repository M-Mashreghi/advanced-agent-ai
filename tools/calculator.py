import ast
import json
import math
import operator

BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
}


def safe_calculate(expression):
    expression = expression.strip().replace("^", "**")
    if len(expression) > 300:
        raise ValueError("Expression is too long.")
    tree = ast.parse(expression, mode="eval")

    def ev(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed.")
        if isinstance(node, ast.BinOp):
            op = BINARY.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported operator.")
            return op(ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            op = UNARY.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported unary operator.")
            return op(ev(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Invalid function.")
            fn = FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ValueError(f"Function '{node.func.id}' is not allowed.")
            return fn(*[ev(arg) for arg in node.args])
        raise ValueError("Invalid mathematical expression.")

    return ev(tree.body)


def expression_from_request(llm, user_request, evidence=""):
    prompt = f"""
Create a safe arithmetic expression that answers the user's calculation request.
If evidence is supplied, use only numerical values supported by that evidence.
Do not invent numbers.
Allowed functions: sqrt, sin, cos, tan, log, log10, abs, round.
Return ONLY JSON: {{"expression":"(10 + 20) / 2"}}
If it cannot be constructed: {{"expression":""}}

User request:
{user_request}

Evidence:
{evidence}
"""
    raw = llm.chat([{"role":"user","content":prompt}], json_mode=True, temperature=0.0)
    return str(json.loads(raw).get("expression", "")).strip()
