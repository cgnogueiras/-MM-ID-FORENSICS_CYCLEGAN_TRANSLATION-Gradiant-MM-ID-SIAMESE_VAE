import os
from pathlib import Path
import json

from tqdm import tqdm


def contains_prohibited_symbols(text):
    for symbol in prohibited_symbols:
        if symbol in text:
            return True
    return False


folder_path = Path("/media/BM/databases/macro_generated_words/input_new/input_new/")
# files = Path(folder_path).glob("*.png")

json_file = "word_num_cap_en_annot_fixed.json"

not_existing_files = set()
with open(json_file, "r") as jsonf:
    data = json.load(jsonf)

prohibited_symbols = ["!", "@", "#", "$", "%", "\\", " ", ","]

fix_data = {k: v.copy() for k, v in data.items()}
fix_data["images"].clear()

for image_data in tqdm(data["images"]):
    file_name = folder_path / image_data["file_name"]
    if not file_name.is_file():
        print(file_name.name)
    else:
        fix_data["images"].append(image_data)
    if len(fix_data["images"]) > 500000:
        break

print("end")
with open("word_num_cap_en_annot_fix_medium.json", "w") as json_file:
    json.dump(fix_data, json_file, indent=4)

# padmaa\-Bold.1.1,padmaa,padmma
# Tlwg Typ_12_655600
# Ubuntu Mon_16_0_False_Truept
