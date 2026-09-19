from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# ============================================================
# TOOL RESULT
# ============================================================

@dataclass
class ToolResult:

    success: bool

    tool: str

    output: Any = None

    error: str | None = None

    metadata: dict | None = None


# ============================================================
# BASE TOOL
# ============================================================

class BaseTool(ABC):

    # Every tool must define these
    name: str = ""
    description: str = ""


    # ========================================================
    # EXECUTE
    # ========================================================

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool and return a ToolResult.
        """
        pass
    