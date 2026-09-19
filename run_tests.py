import subprocess
import sys

from base_tool import BaseTool, ToolResult


class RunTestsTool(BaseTool):

    name = "run_tests"

    description = (
        "Run a Python test suite and return the results."
    )


    def execute(
        self,
        path: str = ".",
        timeout: int = 120
    ) -> ToolResult:

        try:

            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    path
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # ------------------------------------------------
            # Tests passed
            # ------------------------------------------------

            if process.returncode == 0:

                return ToolResult(
                    success=True,
                    tool=self.name,
                    output=process.stdout,
                    metadata={
                        "path": path,
                        "return_code": process.returncode,
                        "stderr": process.stderr
                    }
                )

            # ------------------------------------------------
            # Tests failed
            # ------------------------------------------------

            return ToolResult(
                success=False,
                tool=self.name,
                output=process.stdout,
                error=process.stderr,
                metadata={
                    "path": path,
                    "return_code": process.returncode
                }
            )

        except subprocess.TimeoutExpired:

            return ToolResult(
                success=False,
                tool=self.name,
                error=(
                    f"Tests exceeded "
                    f"{timeout} seconds."
                )
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e)
            )