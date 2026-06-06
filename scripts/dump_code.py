import os
import re
import argparse
from pathlib import Path

# Pre-defined presets
PRESETS = {
    "core": ['src/exozippy'],
    "all": ['src', 'examples', 'tests']
}

# Folders to completely ignore during normal sweeps
EXCLUDE_DIRS = {
    '.git', '.venv', '__pycache__', '.pytest_cache',
    'build', 'dist', 'runs', 'docs', 'data',
    'evolutionary_model'
}


def parse_traceback(tb_file, project_root):
    """Parses a traceback text file and extracts project-relative file paths."""
    paths = set()
    pattern = re.compile(r'File "([^"]+)"')

    with open(tb_file, 'r', encoding='utf-8') as f:
        content = f.read()

    for match in pattern.finditer(content):
        raw_path = match.group(1)
        p = Path(raw_path)

        # Resolve to absolute path if it isn't already
        if not p.is_absolute():
            p = (project_root / p).resolve()

        # Keep it only if it belongs to our project (filters out PyMC, IPython, etc.)
        try:
            rel_path = p.relative_to(project_root)
            paths.add(rel_path)
        except ValueError:
            continue

    return paths


def get_smart_files(traceback_file, project_root):
    """Gathers the Core DNA + Only the components/tests in the traceback."""
    files_to_dump = set()

    # 1. ALWAYS INCLUDE CORE DNA
    # Files directly in src/exozippy/ (run.py, config.py, system.py)
    files_to_dump.update(project_root.glob("src/exozippy/*.py"))
    files_to_dump.update(project_root.glob("src/exozippy/*.yaml"))

    # Files directly in src/exozippy/components/ (component.py, parameter.py)
    files_to_dump.update(project_root.glob("src/exozippy/components/*.py"))
    files_to_dump.update(project_root.glob("src/exozippy/components/*.yaml"))

    # 2. ADD TRACEBACK TARGETS
    tb_paths = parse_traceback(traceback_file, project_root)

    for rel_path in tb_paths:
        parts = rel_path.parts

        # If the traceback touched a specific component subfolder (e.g. components/planet/)
        # We want to grab that whole subfolder so the LLM has its physics and defaults.
        if len(parts) >= 4 and parts[:3] == ('src', 'exozippy', 'components'):
            comp_name = parts[3]
            comp_dir = project_root / 'src' / 'exozippy' / 'components' / comp_name

            if comp_dir.is_dir():
                files_to_dump.update(comp_dir.glob("*.py"))
                files_to_dump.update(comp_dir.glob("*.yaml"))
        else:
            # Otherwise, just add the specific file that crashed (e.g., tests/test_foo.py)
            specific_file = project_root / rel_path
            if specific_file.exists() and specific_file.suffix in ['.py', '.yaml']:
                files_to_dump.add(specific_file)

    return sorted(list(files_to_dump))


def get_preset_files(include_dirs, project_root):
    """Standard os.walk sweep for preset directories."""
    files_to_dump = set()
    for target_dir in include_dirs:
        full_target_path = project_root / target_dir

        if not full_target_path.exists():
            continue

        for root, dirs, files in os.walk(full_target_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                if file.endswith(('.py', '.yaml')):
                    files_to_dump.add(Path(root) / file)

    return sorted(list(files_to_dump))


def dump_code(output_file, target_files, project_root):
    out_path = project_root / output_file

    # Don't dump the dump file into the dump file
    target_files = [f for f in target_files if f != out_path]

    print(f"Dumping {len(target_files)} files to {out_path}...")

    with open(out_path, 'w', encoding='utf-8') as out:
        for file_path in target_files:
            rel_path = file_path.relative_to(project_root)
            out.write(f"\n--- FILE: {rel_path} ---\n")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Enumerate tracks the TRUE line number in the source file
                    for i, line in enumerate(f, 1):
                        # Still strip empty lines to save LLM tokens
                        if line.strip():
                            out.write(f"{i:4d} | {line}")
            except Exception as e:
                out.write(f"Error reading file: {e}\n")
            out.write("\n")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump code for LLM context.")
    parser.add_argument("-o", "--output", default="repo_dump.txt")
    parser.add_argument("-p", "--preset", choices=PRESETS.keys(), default="core")
    parser.add_argument("-t", "--traceback", help="Path to a text file containing the traceback")

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    if args.traceback:
        print(f"Smart Mode: Filtering by traceback in {args.traceback}")
        tb_file = Path(args.traceback)
        if not tb_file.exists():
            print(f"Error: Traceback file {tb_file} not found.")
            exit(1)
        target_files = get_smart_files(tb_file, project_root)
    else:
        print(f"Preset Mode: {args.preset}")
        target_files = get_preset_files(PRESETS[args.preset], project_root)

    dump_code(args.output, target_files, project_root)