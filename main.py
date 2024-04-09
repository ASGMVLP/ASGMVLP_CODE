import datetime
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torchvision
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from transformers import AutoTokenizer, AutoModel

from datasets.data_module import DataModule
from datasets.pretrain_dataset import MultimodalPretrainingDataset, multimodal_collate_fn
from models.encoder import BertEncoder, ImageEncoder
from models.irl_model import IRLModel


class ASG(LightningModule):
    """Pytorch lightning implementation of ASG"""

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 patch_size: int = 14,
                 num_region: int = 23,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.num_region = num_region

        self.irl = IRLModel()
        self.text_list = ['pleural effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube',
                          'consolidation', 'enlarged cardiomediastinum', 'tip', 'pneumonia', 'line', 'cardiomegaly',
                          'fracture', 'calcification', 'device', 'engorgement', 'nodule', 'wire', 'pacemaker',
                          'pleural thicken', 'marking', 'scar', 'hyperinflate', 'blunt', 'collapse', 'emphysema',
                          'aerate', 'mass', 'infiltration', 'obscure', 'deformity', 'hernia', 'drainage', 'distention',
                          'shift', 'stent', 'lesion', 'hardware', 'dilation', 'aspiration']
        self.text_encoder = self._get_bert_basemodel("emilyalsentzer/Bio_ClinicalBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=False)

    def get_text_features(self, model, text_list, tokenizer, device, max_length):
        target_tokenizer = tokenizer(text_list, padding="max_length", truncation=True, max_length=max_length,
                                     return_tensors="pt").to(device)
        text_features = model(input_ids=target_tokenizer["input_ids"],
                              attention_mask=target_tokenizer["attention_mask"])
        text_features = text_features.last_hidden_state[:, 0, :]
        return text_features

    def _get_bert_basemodel(self, bert_model_name):
        try:
            model = AutoModel.from_pretrained(bert_model_name, add_pooling_layer=False)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers library")

        for param in model.parameters():
            param.requires_grad = False

        return model

    def forward(self, batch, batch_idx, split="train"):
        img_feat_q, patch_feat_q = self.img_encoder_q(batch["imgs"])
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        bs = img_feat_q.shape[0]

        # region_feature 
        bbox = batch["regions"]
        patch_emb_q = patch_emb_q.transpose(1, 2).reshape(bs, self.emb_dim, self.patch_size, self.patch_size)
        roi_feature = torchvision.ops.roi_align(patch_emb_q, bbox.reshape(bs * self.num_region, 5), (1, 1), 1.0)
        roi_feature = roi_feature.reshape(bs, self.num_region, self.emb_dim)

        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        report_feat_q, word_feat_q = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        # report_feature
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        # sent_feature 
        sent_index = batch["sent_indexs"]
        valid_index = sent_index != -1
        sent_index_mask = torch.where(valid_index, sent_index, 0)
        sen_emb_q = torch.gather(word_emb_q.unsqueeze(1).expand(-1, self.num_region, -1, -1), 2,
                                 sent_index_mask.unsqueeze(-1).expand(-1, -1, -1, self.emb_dim))
        sen_emb_q = sen_emb_q * valid_index.unsqueeze(-1)
        sen_emb_q = sen_emb_q.sum(dim=2) / valid_index.sum(dim=2, keepdim=True).clamp(min=1)

        # For Image-Report Alignment
        ################################ loss_ira ################################
        labels = torch.arange(bs, device=self.device)
        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss_ira0 = F.cross_entropy(scores, labels)
        loss_ira1 = F.cross_entropy(scores1, labels)
        loss_ira = loss_ira0 + loss_ira1
        ################################ loss_ira ################################

        # For Anatomical Region-Sentence Alignment
        ################################ loss_arsa ################################
        count_matrix = batch["count_matrixs"]
        re_labels = batch["count_labels"]

        scores_re = roi_feature.bmm(sen_emb_q.transpose(1, 2))
        scores_re /= self.hparams.softmax_temperature
        scores_re = scores_re + count_matrix

        loss_arsa0 = F.cross_entropy(scores_re.reshape(-1, self.num_region), re_labels.reshape(-1), ignore_index=-100)
        loss_arsa1 = F.cross_entropy(scores_re.transpose(1, 2).reshape(-1, self.num_region), re_labels.reshape(-1),
                                     ignore_index=-100)
        loss_arsa = loss_arsa0 + loss_arsa1
        ################################ loss_arsa ################################

        # For Internal Representation Learning
        ################################ loss_bce ################################
        text_features = self.get_text_features(self.text_encoder, self.text_list, self.tokenizer, self.device,
                                               max_length=128)
        pred_class_image = self.irl(patch_feat_q, text_features)

        # to get cls_label
        cls_label = batch["labels"]
        cls_label[cls_label == -1] = 0
        loss_bce = F.binary_cross_entropy_with_logits(pred_class_image.reshape(bs, 40),
                                                      cls_label.reshape(bs, 40).float())
        ################################ loss_bce ################################

        # For External Representation Learning
        ################################ loss_soft ################################
        cls_label_norms = F.normalize(cls_label.to(dtype=torch.float32), dim=-1)
        soft_label = torch.mm(cls_label_norms, cls_label_norms.transpose(0, 1))
        hard_label = torch.eye(bs, device=self.device)
        final_label = F.normalize(0.8 * hard_label + 0.2 * soft_label + 1e-6, p=1, dim=-1)

        kl_for = F.kl_div(F.log_softmax(scores, dim=-1), final_label, reduction="batchmean")
        kl_back = F.kl_div(torch.log(final_label), F.softmax(scores, dim=-1), reduction="batchmean")
        loss_soft0 = (kl_for + kl_back) / 2

        kl1_for = F.kl_div(F.log_softmax(scores1, dim=-1), final_label, reduction="batchmean")
        kl1_back = F.kl_div(torch.log(final_label), F.softmax(scores1, dim=-1), reduction="batchmean")
        loss_soft1 = (kl1_for + kl1_back) / 2

        loss_soft = (loss_soft0 + loss_soft1) / 2
        ################################ loss_soft ################################

        # compute retrieval accuracy top top1/top5
        i2t_acc1, i2t_acc5 = self.precision_at_k(scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return loss_ira, loss_arsa, loss_bce, loss_soft, acc1, acc5

    def training_step(self, batch, batch_idx):
        loss_ira, loss_arsa, loss_bce, loss_soft, acc1, acc5 = self(batch, batch_idx, "train")
        loss = loss_ira + loss_arsa + 5 * loss_bce + loss_soft

        log = {
            "train_loss_ira": loss_ira,
            "train_loss_arsa": loss_arsa,
            "train_loss_bce": loss_bce,
            "train_loss_soft": loss_soft,
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss_ira, loss_arsa, loss_bce, loss_soft, acc1, acc5 = self(batch, batch_idx, "valid")
        loss = loss_ira + loss_arsa + 5 * loss_bce + loss_soft

        log = {
            "val_loss_ira": loss_ira,
            "val_loss_arsa": loss_arsa,
            "val_loss_bce": loss_bce,
            "val_loss_soft": loss_soft,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        """ Compute the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=4e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--data_dir", type=str, default="data")
        parser.add_argument("--output_dir", type=str, default="outputs")

        return parser

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_devices)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


def cli_main():
    parser = ArgumentParser()
    parser = ASG.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            args.data_dir, args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = ASG(**args.__dict__)

    # get current time
    now = datetime.datetime.now()
    exp_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join(args.output_dir, f"{exp_name}")
    ckpt_dir = os.path.join(output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")
    ]
    logger_dir = output_dir
    os.makedirs(logger_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=logger_dir, name="logs", version="")
    csv_logger = CSVLogger(save_dir=logger_dir, name="logs", version="")
    wandb_logger = WandbLogger(project="ASG", save_dir=logger_dir, name=exp_name)

    trainer = Trainer(
        strategy="ddp",
        max_epochs=args.max_epochs,
        deterministic=args.deterministic,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger, wandb_logger]
    )

    model.training_steps = model.num_training_steps(trainer, datamodule)
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
