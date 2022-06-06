from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition


def BBTerminalCondition(N_STEPS: int):
    return [GoalScoredCondition(), TimeoutCondition(N_STEPS)]