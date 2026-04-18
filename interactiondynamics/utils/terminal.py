class TermColor:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    HOT_PINK = "\033[38;5;205m"
    SOFT_PINK = "\033[38;5;218m"
    PEACH = "\033[38;5;215m"
    ORANGE = "\033[38;5;208m"
    GOLD = "\033[38;5;222m"


def color_text(text: str, *styles: str) -> str:
    return "".join(styles) + text + TermColor.RESET


def print_finished_line(text: str) -> None:
    print(color_text(text, TermColor.BOLD, TermColor.HOT_PINK))


def print_analysis_line(text: str) -> None:
    print(color_text(text, TermColor.SOFT_PINK))


def print_top_runs_header(text: str) -> None:
    print(color_text(text, TermColor.BOLD, TermColor.ORANGE))


def print_warning_line(text: str) -> None:
    print(color_text(text, TermColor.PEACH))


def print_highlight_line(text: str) -> None:
    print(color_text(text, TermColor.BOLD, TermColor.GOLD))