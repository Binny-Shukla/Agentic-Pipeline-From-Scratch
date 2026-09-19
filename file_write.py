from pathlib import Path

from base_tool import BaseTool, ToolResult


class FileWriteTool(BaseTool):

    name = "file_write"

    description = "Create or overwrite a text file."


    def execute(
        self,
        path: str,
        content: str
    ) -> ToolResult:

        try:

            file_path = Path(path)

            # ------------------------------------------------
            # Create parent directory if necessary
            # ------------------------------------------------

            file_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Write file
            # ------------------------------------------------

            file_path.write_text(
                content,
                encoding="utf-8"
            )

            return ToolResult(
                success=True,
                tool=self.name,
                output=f"File written successfully: {path}",
                metadata={
                    "path": str(file_path),
                    "size": file_path.stat().st_size
                }
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e)
            )


if __name__ == "__main__":

    FileWriteTool()
