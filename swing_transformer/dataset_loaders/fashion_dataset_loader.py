import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.io import read_image
import torchvision.transforms as transforms
from dotenv import load_dotenv
import os

assert load_dotenv(Path(os.path.dirname(__file__)) / "../../.env"), "No .env file found cant load data"
DATA_PATH = os.getenv("DATA_PATH")


class KaggleFashionDataset(Dataset):
    """
    dataset can be downloaded from: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
    """
    def __init__(self, transform_type=None, label="id"):

        # load images dara-frame
        self._data_path = Path(DATA_PATH) / "fashion-dataset"
        self._images = pd.read_csv(self._data_path / "images.csv")
        # set image_id as index
        self._images["image_id"] = self._images.filename.apply(lambda f_name: Path(f_name).stem)
        self._images.set_index("image_id", drop=True, inplace=True)
        self._images.index.name = None

        # load annotations data-frame
        cols = ['id', 'gender', 'masterCategory', 'subCategory', 'articleType',
                'baseColour', 'season', 'year', 'usage', 'productDisplayName']
        self._styles = pd.read_csv(self._data_path / "styles.csv", usecols=cols,
                                       dtype=dict.fromkeys(cols, "str"))
        # set image_id as index - consider change to another string
        self._idx2label = self._styles.subCategory.unique()
        self._label2idx = dict(zip(self._idx2label, np.arange(len(self._idx2label))))
        self._styles['label'] = self._styles.subCategory.map(self._label2idx)

        # set transform - CLIP transforms are applied in the model
        self._transform_type = transform_type if transform_type is not None else "NO_TRANSFORM"
        self._set_transform(transform_type=self._transform_type)
        self._label = label

    def _set_transform(self, transform_type="clip"):
        if transform_type == "resize-norm":
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        else:
            self._transform = None

    def __len__(self):
        return len(self._styles)

    def __getitem__(self, idx):
        # get image data
        img_data = self._styles.iloc[idx]
        img_id = img_data["id"]
        img_label = img_data[self._label]  # TODO: clean text
        img_name = self._images.loc[img_id].filename

        # load image
        img_path = self._data_path / "images" / img_name
        try:
            image = read_image(img_path.as_posix())         # load to tensor
        except:
            # load black image
            image = np.zeros((3, 80, 60), dtype=np.uint8)
        # image = Image.open(img_path.as_posix())       # load to PIL image

        # apply transform
        if self._transform is not None:
            image = self._transform(image)

        return image, img_label


def group_images(batch):
    images, labels = zip(*batch)
    # get shape of each image and find max shape to be used for padding
    shapes = np.vstack([im.shape for im in images])
    max_channels, max_h, max_w = shapes.max(axis=0)
    # pad images to max shape
    images = np.stack([
        np.pad(im, ((0, max_channels - im.shape[0]),
                    (0, max_h -  im.shape[1]),
                    (0, max_w  -  im.shape[2])),
               mode='constant', constant_values=0) for im in images
    ])
    return torch.Tensor(images), labels  # TODO: convert to tensor

# def collate_fn(batch):
#     return batch


if __name__ == '__main__':
    ds = KaggleFashionDataset()
    dl = DataLoader(ds,
                    num_workers=4,
                    batch_size=16,
                    collate_fn=group_images)

    for i, (img, label) in enumerate(dl):
        print(i, img.shape, label)
        break
