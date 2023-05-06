import traceback


def clean_traceback(traceback: str):
    lines = traceback.split("\n")
    new_lines = []
    for line in lines:
        new_line = line.split("discord-tron-client/")[-1].strip()
        new_lines.append(new_line)
    new_string = "\n".join(new_lines)
    return new_string
