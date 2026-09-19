from pathlib import Path

from base_tool import BaseTool, ToolResult


class FileSearchTool(BaseTool):

    name = "file_search"

    description = "Search for files matching a pattern."


    def execute(
        self,
        path: str = ".",
        pattern: str = "*"
    ) -> ToolResult:

        try:

            root = Path(path)

            if not root.exists():

                return ToolResult(
                    success=False,
                    tool=self.name,
                    error=f"Path does not exist: {path}"
                )

            if not root.is_dir():

                return ToolResult(
                    success=False,
                    tool=self.name,
                    error=f"Path is not a directory: {path}"
                )

            matches = []

            for item in root.rglob(pattern):

                if item.is_file():

                    matches.append(str(item))

            return ToolResult(
                success=True,
                tool=self.name,
                output=matches,
                metadata={
                    "path": str(root),
                    "pattern": pattern,
                    "count": len(matches)
                }
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e)
            )


if __name__ == "__main__":

    FileSearchTool()
