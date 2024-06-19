"""Testing `serenityff.charge.utils.io`."""

from pathlib import Path

from serenityff.charge.utils import command_to_shell_file


def test_command_to_shell_file(tmp_path: Path) -> None:
    """Test `command_to_shell_file`.

    Args:
        tmp_path (Path): pytest magic for temporary files.
    """
    file = tmp_path / "test.sh"
    command_to_shell_file("echo Hello World", file.as_posix())
    assert file.exists()
    assert file.read_text() == "#!/bin/bash\n\necho Hello World"
