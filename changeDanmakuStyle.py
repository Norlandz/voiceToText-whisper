import os
import re
import sys


def modify_ass_style(file_path):
    """Modifies the font size and transparency of a specific style in an ASS file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    modified_lines = []
    style_modified = False

    for line in lines:
        if line.strip().startswith("Style: "):
            # Extract existing parts and change what is needed
            parts = line.strip().split(", ")

            # Change Fontsize
            # parts[2] = "16"
            parts[2] = "26"

            # Change Transparency (Alpha Channel of the Primary Colour)
            # ASS uses a &HBBGGRR format, transparency is in the first BB bytes
            # &H00FFFFFF = completely opaque white
            # &H80FFFFFF = 50% transparency white
            # &H33FFFFFF = ~20% transparency white
            # &H00RRGGBB - Completely Opaque
            # &HFFRRGGBB - Completely Transparent

            # Apply 20% transparency
            parts[3] = "&H33FFFFFF"

            modified_line = ", ".join(parts) + "\n"
            modified_lines.append(modified_line)
            style_modified = True
        else:
            modified_lines.append(line)

    # Only overwrite the file if there was a modification.
    if style_modified:
        with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
            f.writelines(modified_lines)
        print(f"Modified: {file_path}")
    else:
        print(f"No Style to Modify: {file_path}")


def process_ass_files(directory):
    """Processes all ASS files in a given directory."""
    for filename in os.listdir(directory):
        if filename.lower().endswith(".ass"):
            file_path = os.path.join(directory, filename)
            modify_ass_style(file_path)

if __name__ == "__main__":
    print(sys.stdout.encoding)
    # ass_directory = input("Enter the directory containing the ASS files: ")
    ass_directory = r"C:\usp\usehpsj\study\Redis"
    if os.path.isdir(ass_directory):
        process_ass_files(ass_directory)
    else:
        print("Error: The input was not a directory")
