import csv
import os
import pickle

import albumentations as A
import cv2
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, data_dir, split="train", imsize=256, crop_size=224, patch_size=14, num_region=23, max_words=112):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.crop_size = crop_size
        self.imsize = imsize
        self.x_scale = patch_size / crop_size
        self.y_scale = patch_size / crop_size
        self.num_region = num_region

        self.transforms = self.get_transforms()

        train_filenames = os.path.join(data_dir, "train_filenames.csv")
        valid_filenames = os.path.join(data_dir, "valid_filenames.csv")
        self.filenames = []
        if split == "train":
            with open(train_filenames, mode='r', newline='') as file:
                reader = csv.reader(file, delimiter=',', quotechar='"')
                for row in reader:
                    self.filenames.append(row[0])
        if split == "valid":
            with open(valid_filenames, mode='r', newline='') as file:
                reader = csv.reader(file, delimiter=',', quotechar='"')
                for row in reader:
                    self.filenames.append(row[0])

        label_path = os.path.join(data_dir, "labels.pickle")
        with open(label_path, "rb") as f:
            self.labels = pickle.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=False)
        self.max_words = max_words

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, key):
        series_sents = self.labels[key]["path2sent"]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)
        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = tokens["attention_mask"][0].sum()

        return tokens, x_len

    def get_region_matrix(self, non_count):
        count_matrix = torch.full((self.num_region, self.num_region), float("-inf"))
        if non_count == 0:
            count_matrix[:1, :1] = 0
            count_matrix[1:, 1:] = 0
        else:
            count_matrix[:non_count, :non_count] = 0
            count_matrix[non_count:, non_count:] = 0
        count_label = torch.arange(self.num_region)
        count_label[non_count:] = -100

        return count_matrix, count_label

    def get_imgs(self, key):
        img_path = os.path.join(self.data_dir, key)
        img = cv2.imread(img_path, 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def get_transforms(self, crop_size: int = 224):
        if self.split == "train":
            data_transforms = A.Compose([
                A.LongestMaxSize(max_size=self.imsize),
                A.PadIfNeeded(min_height=self.imsize, min_width=self.imsize, border_mode=cv2.BORDER_CONSTANT),
                A.RandomCrop(width=crop_size, height=crop_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']))
        else:
            data_transforms = A.Compose([
                A.LongestMaxSize(max_size=self.imsize),
                A.PadIfNeeded(min_height=self.imsize, min_width=self.imsize, border_mode=cv2.BORDER_CONSTANT),
                A.CenterCrop(width=crop_size, height=crop_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']))

        return data_transforms

    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        imgs = self.get_imgs(key)
        label = torch.as_tensor(self.labels[key]["labels"])
        region = torch.as_tensor(self.labels[key]["region"])
        sent_index = torch.as_tensor(self.labels[key]["sent_index"])
        if len(region) == 0:
            len_region_pos = 0
        else:
            len_region_pos = torch.sum(region[:, 0] != -1)
        transformed = self.transforms(image=imgs, bboxes=region[:len_region_pos].tolist(),
                                      labels=torch.arange(self.num_region)[:len_region_pos])
        imgs = transformed["image"]
        idx_select = torch.as_tensor(transformed["labels"])
        if len(transformed["bboxes"]) == 0:
            region = torch.zeros((0, 4), dtype=torch.float32)
            sent_index = torch.zeros((0, 110), dtype=torch.int64)
        else:
            region = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            region = region * torch.tensor([self.x_scale, self.y_scale, self.x_scale, self.y_scale])
            sent_index = sent_index[idx_select]

        non_count = len(region)
        region = torch.nn.functional.pad(region, (0, 0, 0, self.num_region - non_count), "constant", -1)
        sent_index = torch.nn.functional.pad(sent_index, (0, 0, 0, self.num_region - non_count), "constant", -1)

        count_matrix, count_label = self.get_region_matrix(non_count)

        return imgs, caps, cap_len, label, sent_index, region, count_matrix, count_label


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, ids, tokens, attention, labels, sent_index, regions, count_matrixs, count_labels, region_ids = [], [], [], [], [], [], [], [], [], [], []
    for i, b in enumerate(batch):
        img, cap, cap_l, label, sent_idx, region, count_matrix, count_label = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        labels.append(label)
        region_ids.append(torch.full_like(region[:, :1], i))
        sent_index.append(sent_idx)
        regions.append(region)
        count_matrixs.append(count_matrix)
        count_labels.append(count_label)
    # stack
    imgs = torch.stack(imgs)
    cap_len = torch.stack(cap_len)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    labels = torch.stack(labels)
    region_ids = torch.stack(region_ids)
    sent_index = torch.stack(sent_index)
    regions = torch.cat((region_ids, torch.stack(regions)), dim=2)
    count_matrixs = torch.stack(count_matrixs)
    count_labels = torch.stack(count_labels)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(cap_len, 0, True)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "labels": labels[sorted_cap_indices],
        "regions": regions[sorted_cap_indices],
        "sent_indexs": sent_index[sorted_cap_indices],
        "count_matrixs": count_matrixs[sorted_cap_indices],
        "count_labels": count_labels[sorted_cap_indices],
    }

    return return_dict
