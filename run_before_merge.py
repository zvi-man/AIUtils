from subprocess import Popen, PIPE
from mypy import api

# Constants
REQUIREMENTS_FILE_NAME = "Requirements.txt"
REQUIREMENTS_CMD = ["pip", "freeze", ">", REQUIREMENTS_FILE_NAME]
# MYPY_CMD = ["--check-untyped-defs", "--ignore-missing-imports", "../KMUtils"]
MYPY_CMD = ["--check-untyped-defs", "../KMUtils"]


def create_requirements_file():
    with Popen(REQUIREMENTS_CMD, stdout=PIPE) as proc:
        assert proc.stdout is not None
        requirements = proc.stdout.read()
    with open(REQUIREMENTS_FILE_NAME, 'wb') as f:
        f.write(requirements)


def run_mypy_test() -> None:
    print("-------- Starting mypy type checking --------")
    result = api.run(MYPY_CMD)

    if result[0]:
        print('\nType checking report:')
        print(result[0])  # stdout

    if result[1]:
        print('\nError report:')
        print(result[1])  # stderr

    print('Exit status:', result[2])
    print("-------- Finished mypy type checking --------")


if __name__ == '__main__':
    create_requirements_file()
    run_mypy_test()
