import os
from pathlib import Path


# This dumps the code in the specified project folders to a single text file.
# Useful for providing context to AI models for debugging/development.

def dump_code(output_file="repo_dump.txt"):
    # 1. Locate the project root (one level up from this script)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    # 2. Folders we want to include
    include_dirs = ['src', 'tests', 'scripts', 'examples_new']

    # 3. Folders or files to strictly ignore
    exclude_dirs = {'.git', '.venv', '__pycache__', '.ipynb_checkpoints', 'build', 'dist'}
    exclude_files = {'.DS_Store', 'repo_dump.txt', 'collect_code.py'}

    print(f"Scanning project root: {project_root}")

    with open(project_root / output_file, 'w', encoding='utf-8') as out:
        for target_dir in include_dirs:
            full_target_path = project_root / target_dir

            if not full_target_path.exists():
                print(f"Skipping {target_dir}: Directory not found.")
                continue

            for root, dirs, files in os.walk(full_target_path):
                # Modifying dirs in-place to skip excluded folders
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for file in files:
                    # We usually only care about .py and maybe .yaml for this repo
                    if (file.endswith('.py') or file.endswith('.yaml')) and file not in exclude_files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, project_root)

                        out.write(f"\n--- FILE: {rel_path} ---\n")
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                out.write(f.read())
                        except Exception as e:
                            out.write(f"Error reading file: {e}\n")
                        out.write("\n")

    print(f"Done! All relevant code collected in {project_root / output_file}")


if __name__ == "__main__":
    dump_code()