from pathlib import Path

from base_tool import BaseTool, ToolResult


class FileListTool(BaseTool):

    name = "file_list"

    description = "List files and directories at a given path."


    def execute(self, path: str = ".") -> ToolResult:

        try:

            directory = Path(path)

            if not directory.exists():

                return ToolResult(
                    success=False,
                    tool=self.name,
                    error=f"Path does not exist: {path}"
                )

            if not directory.is_dir():

                return ToolResult(
                    success=False,
                    tool=self.name,
                    error=f"Path is not a directory: {path}"
                )

            entries = []

            for item in directory.iterdir():

                entries.append({
                    "name": item.name,
                    "type": "directory"
                    if item.is_dir()
                    else "file"
                })

            return ToolResult(
                success=True,
                tool=self.name,
                output=entries,
                metadata={
                    "path": str(directory),
                    "count": len(entries)
                }
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e)
            )


if __name__ == "__main__":

    FileListTool()
