Development Setup & Guidelines

Welcome to the project! This guide outlines the tools and standards we use to keep our codebase clean, consistent, and reliable. We use Poetry for dependency management and pre-commit to automate our style and quality checks.

# 1. Dependency Management with Poetry

We use Poetry to manage dependencies, virtual environments, and packaging. It replaces pip and requirements.txt with a more robust pyproject.toml file.
Installation & Setup

If you don't have Poetry installed, install it globally using the official installer:
Bash

curl -sSL https://install.python-poetry.org | python3 -

Once installed, navigate to the project root and install the dependencies. This will automatically create a virtual environment for you:
Bash

poetry install

Common Poetry Commands (Bash):

Activate the virtual environment (Bash):

    poetry shell

Run a command inside the environment (without activating it):

    poetry run python main.py

Add a new production dependency:

    poetry add requests

Add a new development dependency:

    poetry add --group dev pytest

Update dependencies:

    poetry update

# 2. Automated Checks with Pre-commit

We use pre-commit to (enforce style guidelines?) and catch common errors before they are committed to the repository. These checks run automatically every time you type git commit.

Installation & Setup

Pre-commit is included in our development dependencies. Once you have run poetry install, you just need to install the git hooks:

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

    UPPER_SNAKE_CASE for module-level constants.

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

AI use is encouraged, but thorough review and testing is essential. Create unit tests that verify/confirm the output for all essential code (see above). Unit testing is especially critical with AI generated code, as it is often tunnel visioned and drops important features.

scripts/dump_code.py will collate the entire repo into a copy/pasteable file for AI review. 