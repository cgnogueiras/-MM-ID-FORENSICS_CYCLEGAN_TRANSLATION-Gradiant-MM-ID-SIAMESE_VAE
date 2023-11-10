import os
from pathlib import Path

folder_path = "/media/BM/databases/macro_generated_words/input/"
files = Path(folder_path).glob("*.png")

fonts_wrong = set()

for file_path in files:
    if "Uroo_" in file_path:
        filename = file_path.name

        new_path = file_path.parent / file_path.stem

        os.rename(file_path, new_path)
        print(f"Renamed '{filename}' to '{new_path}'")


# for file_path in files:
#     filename = file_path.name
#     if ".pdf" in filename:
#         new_filename = (
#             "_".join(filename.split(".pdf")[0].split("_")[:-1]) + filename.split(".pdf")[1]
#         )

#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_filename)

#         try:
#             os.rename(old_path, new_path)
#         except Exception:
#             print("skipping " + new_path)
#             fonts_wrong.add(new_filename.split("_")[0])
#             continue

#         print(f"Renamed '{filename}' to '{new_filename}'")

print("File renaming completed.")
