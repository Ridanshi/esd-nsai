class FSMState:
    EVIDENCE_SPARSE = 0
    HYPOTHESIS_FORMING = 1
    CERTAINTY_BUILDING = 2
    DIAGNOSTIC_TENSION = 3
    RESOLVED = 4


class DiagnosticFSM:
    def traverse(self, certainty_scores: dict, conflict_load: float) -> int:
        """
        Traverses 5-state FSM deterministically. Returns final state as int.
        States advance forward only — never backward.
        """
        top_certainty = max(certainty_scores.values()) if certainty_scores else 0.0
        diseases_above_threshold = sum(
            1 for s in certainty_scores.values() if s > 0.4
        )

        state = FSMState.EVIDENCE_SPARSE

        if top_certainty > 0.1:
            state = FSMState.HYPOTHESIS_FORMING

        if state >= FSMState.HYPOTHESIS_FORMING and top_certainty > 0.4:
            state = FSMState.CERTAINTY_BUILDING

        if state >= FSMState.CERTAINTY_BUILDING and (
            conflict_load > 0.3 or diseases_above_threshold >= 2
        ):
            state = FSMState.DIAGNOSTIC_TENSION

        # Overrides DIAGNOSTIC_TENSION if certainty is strong enough
        if (top_certainty > 0.65 and conflict_load < 0.25) or top_certainty > 0.80:
            state = FSMState.RESOLVED

        return int(state)
