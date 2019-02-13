#!/usr/bin/env python
import os
import re
import argparse

# Define the header template
NEW_HEADER_TEMPLATE = """\
# ******************************************************************************
# This file is part of dlplay
# 
# Copyright (C) Luigi Freda <luigi dot freda at gmail dot com>
# 
# dlplay is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dlplay is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dlplay. If not, see <http://www.gnu.org/licenses/>.
# ******************************************************************************
"""

# Define a regex pattern to identify the existing header
HEADER_REGEX = re.compile(
    r"^# \*+\s*\n"  # Match the start of the block with '# ' and one or more asterisks
    r"(.*?)"  # Match any content in between (non-greedy)
    r"^# \*+\s*\n",  # Match the end of the block with '# ' and one or more asterisks
    re.MULTILINE
    | re.DOTALL,  # MULTILINE to handle line starts and DOTALL to include newlines in content
)

# List of folder names to exclude
EXCLUDED_FOLDERS = {
    "build",
    "lib",
    "third_party",
    "thirdparty",
    "Thirdparty",
    "external",
    ".git",
}

# List of file extensions to include
INCLUDED_EXTENSIONS = {".py"}


def update_header_in_file(filepath):
    # Read the file content
    with open(filepath, "r") as file:
        content = file.read()

    # Check if an existing header matches the template and replace it
    if HEADER_REGEX.search(content):
        content = HEADER_REGEX.sub(NEW_HEADER_TEMPLATE, content)
        print(f"Updated header in {filepath}.")
    else:
        # If no header exists, prepend the new header
        content = NEW_HEADER_TEMPLATE + "\n" + content
        print(f"Added new header to {filepath}.")

    # Write the updated content back to the file
    with open(filepath, "w") as file:
        file.write(content)


def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        # Skip excluded folders
        dirs[:] = [d for d in dirs if d not in EXCLUDED_FOLDERS]

        for file in files:
            # Process only files with included extensions
            if any(file.endswith(ext) for ext in INCLUDED_EXTENSIONS):
                update_header_in_file(os.path.join(root, file))


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(
        description="Add or update header in source files."
    )
    parser.add_argument("directory", type=str, help="Directory to process")
    args = parser.parse_args()

    # Start processing the directory
    process_directory(args.directory)

    print("Done.")
