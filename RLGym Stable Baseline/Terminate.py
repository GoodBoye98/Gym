from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition


def BBTerminalCondition():
    return GoalScoredCondition()