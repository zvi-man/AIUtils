from subprocess import Popen, PIPE
import subprocess

# Constants
REQUIREMENTS_FILE_NAME = "Requirements.txt"
REQUIREMENTS_CMD = ["pip", "freeze", ">", REQUIREMENTS_FILE_NAME]


def create_requirements_file() -> None:
    with Popen(REQUIREMENTS_CMD, stdout=PIPE) as proc:
        requirements = proc.stdout.read()
    with open(REQUIREMENTS_FILE_NAME, 'wb') as f:
        f.write(requirements)


if __name__ == '__main__':
    create_requirements_file()
