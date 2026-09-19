import subprocess
import sys

from base_tool import BaseTool, ToolResult


class PipInstallTool(BaseTool):

    name = "pip_install"

    description = (
        "Install Python packages using pip."
    )


    def execute(
        self,
        package: str,
        timeout: int = 120
    ) -> ToolResult:

        if not package or not package.strip():

            return ToolResult(
                success=False,
                tool=self.name,
                error="Package name cannot be empty."
            )

        package = package.strip()

        try:

            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    package
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if process.returncode == 0:

                return ToolResult(
                    success=True,
                    tool=self.name,
                    output=process.stdout,
                    metadata={
                        "package": package,
                        "return_code": process.returncode
                    }
                )

            return ToolResult(
                success=False,
                tool=self.name,
                output=process.stdout,
                error=process.stderr,
                metadata={
                    "package": package,
                    "return_code": process.returncode
                }
            )

        except subprocess.TimeoutExpired:

            return ToolResult(
                success=False,
                tool=self.name,
                error=(
                    f"pip installation exceeded "
                    f"{timeout} seconds."
                )
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e)
            )