import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, data_dir, batch_size, num_workers):
        super().__init__()

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = self.dataset(self.data_dir, split="train")
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        dataset = self.dataset(self.data_dir, split="valid")
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        dataset = self.dataset(self.data_dir, split="test")
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
