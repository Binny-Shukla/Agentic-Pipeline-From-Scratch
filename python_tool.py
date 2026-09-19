import subprocess
import sys
import tempfile
from pathlib import Path

from base_tool import BaseTool, ToolResult


class PythonTool(BaseTool):

    name = "python"

    description = (
        "Execute Python code in a controlled subprocess "
        "and return its output."
    )

    def execute(
        self,
        code: str,
        timeout: int = 30
    ) -> ToolResult:

        if not code or not code.strip():

            return ToolResult(
                success=False,
                tool=self.name,
                error="Python code cannot be empty."
            )

        temp_file = None

        try:

            # ------------------------------------------------
            # Create temporary Python script
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                encoding="utf-8",
                delete=False
            ) as file:

                file.write(code)
                temp_file = Path(file.name)


            # ------------------------------------------------
            # Execute in separate Python process
            # ------------------------------------------------

            process = subprocess.run(
                [
                    sys.executable,
                    str(temp_file)
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )


            # ------------------------------------------------
            # Successful execution
            # ------------------------------------------------

            if process.returncode == 0:

                return ToolResult(
                    success=True,
                    tool=self.name,
                    output=process.stdout,
                    metadata={
                        "return_code": process.returncode,
                        "stderr": process.stderr
                    }
                )


            # ------------------------------------------------
            # Python execution failed
            # ------------------------------------------------

            return ToolResult(
                success=False,
                tool=self.name,
                output=process.stdout,
                error=process.stderr,
                metadata={
                    "return_code": process.returncode
                }
            )


        except subprocess.TimeoutExpired:

            return ToolResult(
                success=False,
                tool=self.name,
                error=(
                    f"Python execution exceeded "
                    f"{timeout} seconds."
                )
            )


        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e)
            )


        finally:

            # ------------------------------------------------
            # Remove temporary script
            # ------------------------------------------------

            if temp_file is not None:

                try:
                    temp_file.unlink(
                        missing_ok=True
                    )

                except Exception:
                    pass