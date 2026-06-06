Development Setup & Guidelines

Welcome to the project! This guide outlines the tools and standards we use to keep our codebase clean, consistent, and reliable. We use Poetry for dependency management and pre-commit to automate our style and quality checks.

# 1. Dependency Management with Poetry

We use Poetry to manage dependencies, virtual environments, and packaging. It replaces pip and requirements.txt with a more robust pyproject.toml file.
Installation & Setup

First, we need to be sure we're on the same version of python (we'll test version-related bugs separately and explicitly):

    pyenv install 3.12.13

You might need to install pyenv first:

MacOS:

    brew install pyenv


If you don't have Poetry installed, install it globally using the official installer:
Bash

    curl -sSL https://install.python-poetry.org | python3 -

Download EXOZIPPy (if you haven't already):

    git clone https://github.com/jdeast/EXOZIPPy.git

Navigate to the project root and install the dependencies. This will automatically create a virtual environment for you:
Bash

    poetry env use python3.12

    poetry install

When you git pull, be sure to install any new dependencies:

    git pull
    poetry update

To run a python script inside the poetry environment (without activating it):

    poetry run python main.py

If you need to add a new dependent package:

    poetry add <package>

To add a new development dependency:

    poetry add --group dev pytest

Update dependencies:

    poetry update

# 2. Automated Checks with Pre-commit

We use pre-commit to (enforce style guidelines?) and catch common errors before they are committed to the repository. These checks run automatically every time you type git commit.

Installation & Setup

Pre-commit is included in our development dependencies. Once you have run "poetry install", you just need to install the git hooks:

    poetry run pre-commit install

You should see a message saying pre-commit installed at .git/hooks/pre-commit.
How it Works

When you commit code, pre-commit will run a series of tools (defined in .pre-commit-config.yaml) over your staged files.

If everything passes: Your commit succeeds.

If a check fails: The commit is aborted.

Common Pre-commit Commands (bash)

Run checks manually on all files:

    poetry run pre-commit run --all-files

Run checks on specific files:

    poetry run pre-commit run --files src/my_script.py

# 3. Style Guidelines

We don't actually do this. Should we?

We follow standard Python conventions to ensure readability and maintainability.
The "Big Three" Formatters

You rarely need to worry about formatting code manually. Our pre-commit hooks will automatically format your code using the following tools:

    Black: The uncompromising code formatter. We use Black's default rules (e.g., 88-character line length, double quotes for strings). If Black formats it, that is the standard.

    isort: Automatically sorts your imports alphabetically and separates them into logical sections (standard library, third-party, first-party).

    Ruff / Flake8: Used for linting to catch unused imports, undefined variables, and stylistic issues that Black doesn't cover.

General Conventions:

    PEP 8: We adhere to PEP 8 standards.

    No Type Hinting

Naming Conventions:

    CamelCase for Classes.

    snake_case for variables, functions, and methods.

    All constants should be defined in/imported from constants.py and use UPPER_SNAKE_CASE.

Docstrings: Write docstrings for all public modules, classes, and functions using the Google Style. Briefly explain what the function does, its arguments (Args:), and what it returns (Returns:).

# 4. Unit testing:

Following python conventions, this test suite
  - uses long, explicit function names to describe the test
  - Has Given/When/Then doc strings
  - follows the AAA (Arrange, Act, Assert) organizational scheme

Developers are forced to pass all unit tests before pushing a commit via git hooks

Anytime a developer fixes a bug, a new unit test following the above convention should be added.
The test should demonstrate failure before the fix and success after the fix.

# 5. AI use:

AI use is encouraged, but thorough review and testing is essential. Create unit tests that verify/confirm the output for all essential code (see above). Unit testing is especially critical with AI generated code, as it is often tunnel visioned and drops important features not relevant to the bug.

scripts/dump_code.py will collate the entire repo into a copy/pasteable file for AI review. Note that, anecdotally, ~1M tokens is suffificient to keep ~5000 lines of code in context sufficient for a deep, logically review. Beyond that, it tends to lose its focus and forget aspects of the code. The current code base is a bit larger than that, so you may wish to filter the repo dump for more targeted, relevant advice. 

Useful prompts: 
    
    here is the code dump. identify any inconsistent use of functions, style, variable names, etc. Identify any AI comments irrelevant to the final code base. Flag obsolete code. Suggest revisions for speed, clarity, standardization, and readability. Identify arbitrary fallbacks designed to make the code run at all costs, when that cost is producing garbage [paste code dump]

    here is the code dump. write unit tests to protect functionality not currently covered [paste code dump with -c all flag, which includes the unit test directory]