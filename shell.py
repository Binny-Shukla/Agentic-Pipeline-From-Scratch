import subprocess

from base_tool import BaseTool, ToolResult


class ShellTool(BaseTool):

    name = "shell"

    description = (
    "Execute an approved command through the operating-system terminal."
    )

    # Commands Cherry is initially allowed to execute.
    ALLOWED_COMMANDS = {
        "dir",
        "echo",
        "type",
        "where",
        "python",
        "git",
    }


    def execute(
        self,
        command: str,
        timeout: int = 30
    ) -> ToolResult:

        if not command or not command.strip():

            return ToolResult(
                success=False,
                tool=self.name,
                error="Command cannot be empty."
            )

        command = command.strip()

        # ----------------------------------------------------
        # Determine executable
        # ----------------------------------------------------

        executable = command.split()[0].lower()

        if executable not in self.ALLOWED_COMMANDS:

            return ToolResult(
                success=False,
                tool=self.name,
                error=(
                    f"Command not allowed: {executable}"
                )
            )

        try:

            process = subprocess.run(
                command,
                shell=True,
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
                        "return_code": process.returncode,
                        "stderr": process.stderr
                    }
                )

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
                    f"Command exceeded "
                    f"{timeout} seconds."
                )
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e)
            )