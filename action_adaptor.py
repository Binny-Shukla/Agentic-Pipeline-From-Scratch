from dataclasses import dataclass


# ============================================================
# ACTION MAP
# ============================================================

ACTION_MAP = {
    0: "inspect",
    1: "prepare",
    2: "build",
    3: "execute",
    4: "test",
    5: "repair",
    6: "observe",
    7: "verify",
    8: "export",
    9: "no_op"
}


# ============================================================
# ACTION CANDIDATE
# ============================================================

@dataclass
class ActionCandidate:

    action_id: int

    action: str

    rank: int


# ============================================================
# ACTION ADAPTER
# ============================================================

class ActionAdapter:

    def __init__(self, action_map=None):

        self.action_map = action_map or ACTION_MAP


    # --------------------------------------------------------
    # Convert action IDs → semantic actions
    # --------------------------------------------------------

    def adapt(self, action_ids):

        candidates = []

        for rank, action_id in enumerate(action_ids, start=1):

            action_id = int(action_id)

            action = self.action_map.get(
                action_id,
                f"action_{action_id}"
            )

            candidates.append(
                ActionCandidate(
                    action_id=action_id,
                    action=action,
                    rank=rank
                )
            )

        return candidates


    # --------------------------------------------------------
    # Convert a single action ID
    # --------------------------------------------------------

    def adapt_one(self, action_id):

        action_id = int(action_id)

        return ActionCandidate(
            action_id=action_id,
            action=self.action_map.get(
                action_id,
                f"action_{action_id}"
            ),
            rank=1
        )

if __name__ == "__main__":

    adapter = ActionAdapter()

    test_actions = [2, 4, 8]

    candidates = adapter.adapt(test_actions)

    for candidate in candidates:
        print(
            f"Rank {candidate.rank}: "
            f"{candidate.action} "
            f"(ID: {candidate.action_id})"
        )
        