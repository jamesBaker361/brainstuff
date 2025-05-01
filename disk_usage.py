import os

def get_dir_size(path="/scratch/jlb638"):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):  # avoid broken symlinks
                total += os.path.getsize(fp)
    return total

size_in_bytes = get_dir_size(".")
print(f"Total size: {size_in_bytes / (1024 ** 2):.2f} MB")
