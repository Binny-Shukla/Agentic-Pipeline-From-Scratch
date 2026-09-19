from pathlib import Path

from base_tool import BaseTool, ToolResult


class FileReadTool(BaseTool):

    name = "file_read"

    description = "Read the contents of a file."


    def execute(self, path: str) -> ToolResult:

        try:

            file_path = Path(path)

            if not file_path.exists():

                return ToolResult(
                    success=False,
                    tool=self.name,
                    error=f"File does not exist: {path}"
                )

            if not file_path.is_file():

                return ToolResult(
                    success=False,
                    tool=self.name,
                    error=f"Path is not a file: {path}"
                )

            content = file_path.read_text(
                encoding="utf-8"
            )

            return ToolResult(
                success=True,
                tool=self.name,
                output=content,
                metadata={
                    "path": str(file_path),
                    "size": file_path.stat().st_size
                }
            )

        except UnicodeDecodeError:

            return ToolResult(
                success=False,
                tool=self.name,
                error=f"File is not valid UTF-8 text: {path}"
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e)
            )


if __name__ == "__main__":

    tool = FileReadTool()
