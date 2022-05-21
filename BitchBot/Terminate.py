from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition, TimeoutCondition


def BBTerminalCondition():
    return TimeoutCondition(256)