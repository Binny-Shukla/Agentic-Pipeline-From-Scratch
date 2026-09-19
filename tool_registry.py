from typing import Dict


class ToolRegistry:

    def __init__(self):

        self.tools: Dict[str, object] = {}


    # ========================================================
    # REGISTER TOOL
    # ========================================================

    def register(self, tool):

        if not hasattr(tool, "name"):

            raise ValueError(
                "Tool must have a 'name' attribute."
            )

        if tool.name in self.tools:

            raise ValueError(
                f"Tool already registered: {tool.name}"
            )

        self.tools[tool.name] = tool


    # ========================================================
    # GET TOOL
    # ========================================================

    def get(self, tool_name: str):

        if tool_name not in self.tools:

            raise KeyError(
                f"Tool not registered: {tool_name}"
            )

        return self.tools[tool_name]


    # ========================================================
    # CHECK TOOL
    # ========================================================

    def has(self, tool_name: str) -> bool:

        return tool_name in self.tools


    # ========================================================
    # LIST TOOLS
    # ========================================================

    def list_tools(self):

        return list(self.tools.keys())


    # ========================================================
    # REMOVE TOOL
    # ========================================================

    def remove(self, tool_name: str):

        if tool_name in self.tools:

            del self.tools[tool_name]


    # ========================================================
    # REGISTER DEFAULT TOOLS
    # ========================================================

    def register_default_tools(self):

        from file_read import FileReadTool
        from file_list import FileListTool
        from file_search import FileSearchTool
        from file_write import FileWriteTool
        from calculator import CalculatorTool
        from python_tool import PythonTool
        from pip_install import PipInstallTool
        from shell import ShellTool
        from run_tests import RunTestsTool

        tools = [

            FileReadTool(),
            FileListTool(),
            FileSearchTool(),
            FileWriteTool(),
            CalculatorTool(),
            PythonTool(),
            PipInstallTool(),
            ShellTool(),
            RunTestsTool()

        ]

        for tool in tools:

            self.register(tool)
            
