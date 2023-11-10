import json


with open("word_num_cap_en_annot.json", "r") as json_file:
    coco_data = json.load(json_file)

category_name_to_id = {}

cont = 0
for cat in coco_data["categories"]:
    cat["id"] = cont
    cat["name"] = "_".join(cat["name"].split(".pdf")[0].split("_")[:-1])
    category_name_to_id[cat["name"]] = cat["id"]
    cont += 1

for image in coco_data["images"]:
    image["category_id"] = category_name_to_id[
        "_".join(image["file_name"].split(".pdf")[0].split("_")[:-1])
    ]

    image["file_name"] = (
        "_".join(image["file_name"].split(".pdf")[0].split("_")[:-1])
        + image["file_name"].split(".pdf")[1]
    )


with open("word_num_cap_en_annot_fixed.json", "w") as json_file:
    json.dump(coco_data, json_file, indent=4)

print("Category IDs have been fixed and saved to 'coco_annotations_fixed.json'.")
