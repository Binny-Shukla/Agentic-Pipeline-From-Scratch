import ast
import operator

from base_tool import BaseTool, ToolResult


class CalculatorTool(BaseTool):

    name = "calculator"

    description = "Safely evaluate basic mathematical expressions."


    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }


    def execute(self, expression: str) -> ToolResult:

        try:

            expression = expression.strip()

            if not expression:

                return ToolResult(
                    success=False,
                    tool=self.name,
                    error="Expression cannot be empty."
                )

            tree = ast.parse(
                expression,
                mode="eval"
            )

            result = self._evaluate(tree.body)

            return ToolResult(
                success=True,
                tool=self.name,
                output=result,
                metadata={
                    "expression": expression
                }
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e)
            )


    def _evaluate(self, node):

        # Numbers
        if isinstance(node, ast.Constant):

            if isinstance(node.value, (int, float)):

                return node.value

            raise ValueError(
                "Only numeric values are allowed."
            )


        # Binary operations
        if isinstance(node, ast.BinOp):

            operator_function = self.OPERATORS.get(
                type(node.op)
            )

            if operator_function is None:

                raise ValueError(
                    "Unsupported operator."
                )

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            return operator_function(
                left,
                right
            )


        # Unary operations
        if isinstance(node, ast.UnaryOp):

            operator_function = self.OPERATORS.get(
                type(node.op)
            )

            if operator_function is None:

                raise ValueError(
                    "Unsupported unary operator."
                )

            operand = self._evaluate(
                node.operand
            )

            return operator_function(
                operand
            )


        raise ValueError(
            "Only arithmetic expressions are supported."
        )