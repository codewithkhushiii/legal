import os

def combine_code_to_markdown(repo_path, output_file):
    # The extensions you want to target
    target_extensions = {'.js', '.py', '.css', '.html'}
    
    # Folders to ignore so the script doesn't read massive dependency files
    ignore_folders = {'.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build', '.next'}

    with open(output_file, 'w', encoding='utf-8') as md:
        md.write(f"# Codebase from `{os.path.abspath(repo_path)}`\n\n")

        # Walk through the directory
        for root, dirs, files in os.walk(repo_path):
            # Exclude ignored folders from the search
            dirs[:] = [d for d in dirs if d not in ignore_folders]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                
                # If the file matches our target extensions
                if ext in target_extensions:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    # Remove the dot for the markdown code block identifier (e.g., .js -> js)
                    lang = ext[1:] 

                    # Write the header and open the code block
                    md.write(f"### File: `{rel_path}`\n\n")
                    md.write(f"```{lang}\n")

                    # Read and write the file contents
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            md.write(f.read())
                    except Exception as e:
                        md.write(f"// Error reading file: {e}")

                    # Close the code block
                    md.write("\n```\n\n---\n\n")

if __name__ == "__main__":
    # Settings: '.' means current directory, or you can paste a specific folder path
    repo_directory = "." 
    output_markdown_file = "my_repository_code.md"

    print(f"Scanning '{repo_directory}'...")
    combine_code_to_markdown(repo_directory, output_markdown_file)
    print(f"Done! Your code has been saved to '{output_markdown_file}'.")