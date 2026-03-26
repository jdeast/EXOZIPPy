import os

# this dumps the code in the entire repo to a single text file
# this is useful for AI-assisted debugging/development

def dump_code(root_dir, output_file="repo_dump.txt"):
    # Folders or files to ignore
    exclude_dirs = {'.git', '.venv', '__pycache__', '.ipynb_checkpoints', 'build', 'dist'}
    exclude_files = {'.DS_Store', 'collect_code.py'}

    with open(output_file, 'w', encoding='utf-8') as out:
        for root, dirs, files in os.walk(root_dir):
            # Modifying dirs in-place to skip excluded folders
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith('.py') and file not in exclude_files:
                    file_path = os.path.join(root, file)
                    # Create a relative path for the header
                    rel_path = os.path.relpath(file_path, root_dir)

                    out.write(f"\n--- FILE: {rel_path} ---\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            out.write(f.read())
                    except Exception as e:
                        out.write(f"Error reading file: {e}\n")
                    out.write("\n")

    print(f"Done! All code collected in {output_file}")


if __name__ == "__main__":
    # Assuming you run this from the project root
    dump_code(".")