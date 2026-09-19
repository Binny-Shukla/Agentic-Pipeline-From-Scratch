from dataclasses import dataclass
from typing import Any
import time


@dataclass
class ToolResult:

    success: bool

    tool: str

    output: Any = None

    error: str | None = None

    execution_time: float = 0.0


class ToolExecutor:

    def __init__(self, tool_registry):

        self.tool_registry = tool_registry


    # ========================================================
    # EXECUTE SINGLE TOOL
    # ========================================================

    def execute(
        self,
        tool_name: str,
        arguments: dict | None = None
    ) -> ToolResult:

        arguments = arguments or {}

        try:

            tool = self.tool_registry.get(
                tool_name
            )

        except KeyError:

            return ToolResult(
                success=False,
                tool=tool_name,
                error=f"Tool not found: {tool_name}"
            )


        start_time = time.perf_counter()

        try:

            output = tool.execute(
                **arguments
            )

            return ToolResult(
                success=output.success,
                tool=tool_name,
                output=output.output,
                error=output.error,
                execution_time=(
                    time.perf_counter()
                    - start_time
                )
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=tool_name,
                error=str(e),
                execution_time=(
                    time.perf_counter()
                    - start_time
                )
            )


    # ========================================================
    # EXECUTE SEQUENCE
    # ========================================================

    def execute_sequence(
        self,
        tools: list[str],
        arguments: dict | None = None
    ):

        arguments = arguments or {}

        results = []

        for tool_name in tools:

            tool_arguments = arguments.get(
                tool_name
            )

            # -----------------------------------------------
            # Skip tools that have no supplied arguments
            # -----------------------------------------------

            if tool_arguments is None:

                continue


            result = self.execute(
                tool_name,
                tool_arguments
            )

            results.append(result)


            # -----------------------------------------------
            # Stop sequence on failure
            # -----------------------------------------------

            if not result.success:

                break

        return results