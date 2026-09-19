# tools/tool_router.py

from dataclasses import dataclass


# ============================================================
# ROUTE
# ============================================================

@dataclass
class ToolRoute:

    action: str

    tools: list[str]


# ============================================================
# TOOL ROUTER
# ============================================================

class ToolRouter:

    def __init__(self):

        self.routes = {

            # ------------------------------------------------
            # Observation
            # ------------------------------------------------

            "inspect": [
                "file_read",
                "file_search",
                "file_list"
            ],

            "observe": [
                "file_read",
                "file_list",
                "git_status"
            ],

            # ------------------------------------------------
            # Preparation
            # ------------------------------------------------

            "prepare": [
                "file_read",
                "file_list",
                "pip_install",
                "python"
            ],

            # ------------------------------------------------
            # Construction
            # ------------------------------------------------

            "build": [
                "file_read",
                "python",
                "file_write"
            ],

            # ------------------------------------------------
            # Execution
            # ------------------------------------------------

            "execute": [
                "shell",
                "python"
            ],

            # ------------------------------------------------
            # Testing
            # ------------------------------------------------

            "test": [
                "run_tests"
            ],

            # ------------------------------------------------
            # Repair
            # ------------------------------------------------

            "repair": [
                "file_read",
                "pip_install",
                "python",
                "file_write",
                "run_tests"
            ],

            # ------------------------------------------------
            # Verification
            # ------------------------------------------------

            "verify": [
                "python",
                "run_tests"
            ],

            # ------------------------------------------------
            # Export
            # ------------------------------------------------

            "export": [
                "file_write"
            ],

            # ------------------------------------------------
            # No operation
            # ------------------------------------------------

            "no_op": []
        }


    # ========================================================
    # ROUTE ACTION
    # ========================================================

    def route(self, action: str) -> ToolRoute:

        action = action.lower().strip()

        if action not in self.routes:

            raise ValueError(
                f"No tool route defined for action: {action}"
            )

        return ToolRoute(
            action=action,
            tools=self.routes[action].copy()
        )


    # ========================================================
    # AVAILABLE ACTIONS
    # ========================================================

    def available_actions(self):

        return list(self.routes.keys())


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    router = ToolRouter()
