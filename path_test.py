from pathlib import Path
import re
import os

# Set your target directory and desired name prefix
directory = Path(os.path.join(os.environ["BRAIN_DATA_DIR"],"models"))
name = "sub_1_res_3"  # e.g., "model", "checkpoint", etc.

# Regular expression to match files like name_123.pth
pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.pth$")

max_e = -1
max_file = None

for file in directory.glob(f"{name}_*.pth"):
    match = pattern.match(file.name)
    if match:
        e = int(match.group(1))
        if e > max_e:
            max_e = e
            max_file = file

print(f"The file with the largest e for name '{name}' is: {max_file}")
