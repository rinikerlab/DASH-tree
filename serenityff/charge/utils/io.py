def command_to_shell_file(command: str, filename: str) -> None:
    """
    Writes a string to a .sh file

    Args:
        command (str): string to be written
        filename (str): path to file
    """
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(command)
    return
