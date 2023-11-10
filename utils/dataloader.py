import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset


class SiameseNetworkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annot_path: str,
        data_dir: str = "",
        transform=None,
    ):
        """Dataloader from dataset.

        Args:
            annot_path (str):
            data_dir (str, optional): Root directory where images are downloaded to. Defaults to None.
            transform (_type_, optional): image tranformation. Defaults to None.
        """

        self.data_dir = data_dir
        self.transform = transform

        with open(annot_path, "r") as f:
            try:
                annot_data = json.load(f)
                image_data = annot_data["images"]

                category_data = annot_data["categories"]

                category_id_to_name = {
                    category["id"]: category["name"] for category in category_data
                }

                for img in image_data:
                    img["category_id_name"] = category_id_to_name[img["category_id"]]

            except json.JSONDecodeError as e:
                raise ValueError("Annotation file is not a valid JSON file.") from e

        # Load annotations
        self.data = image_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img0 = Image.open(
            os.path.join(
                self.data_dir, "input_new", "input_new", self.data[index]["file_name"]
            )
        )
        target0 = Image.open(
            os.path.join(
                self.data_dir,
                "output",
                f"{self.data[index]['category_id_name']}.png",
            )
        )

        positive = torch.randint(0, 2, size=(1,)).item()
        if positive:
            while True:
                idx2 = torch.randint(0, len(self.data), size=(1,)).item()
                if self.data[idx2]["category_id"] == self.data[index]["category_id"]:
                    break
        else:
            while True:
                idx2 = torch.randint(0, len(self.data), size=(1,)).item()
                if self.data[idx2]["category_id"] != self.data[index]["category_id"]:
                    break

        img1 = Image.open(
            os.path.join(
                self.data_dir, "input_new", "input_new", self.data[idx2]["file_name"]
            )
        )
        target1 = Image.open(
            os.path.join(
                self.data_dir,
                "output",
                f"{self.data[idx2]['category_id_name']}.png",
            )
        )

        if self.transform is not None:
            img0 = self.transform(img0)
            target0 = self.transform(target0)
            img1 = self.transform(img1)
            target1 = self.transform(target1)

        label = self.data[index]["category_id"] == self.data[idx2]["category_id"]

        # # For tranforms that modify images visually
        # concatenated_img = T.ToPILImage()(torch.cat((T.ToTensor()(img0), T.ToTensor()(img1)), dim=2))
        # if self.transform is not None:
        #     concatenated_img = self.transform(concatenated_img)
        # img0, img1 = torch.split( concatenated_img, split_size_or_sections=img0.shape[-1], dim=2,)

        return img0, img1, target0, target1, label

    def __len__(self):
        return len(self.data)
