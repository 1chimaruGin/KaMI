"""
KaMi
1. 
"""
import os
import shutil
import wandb
import torch
import torch.nn as nn

from glob import glob
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from utils.utils import AverageMeter


class KaMi(nn.Module):
    def __init__(self, **kwargs):
        super(KaMi, self).__init__()
        try:
            cfg = self.fetch_config(kwargs.pop("config"))
        except KeyError:
            print("Config Requried!!")
        wandb.init(config=cfg, project=cfg["project"], name=cfg["model"] or None)
        self.__dict__.update({k: v for k, v in wandb.config.items()})

        try:
            self.loss = self.loss_fn() or eval(f"nn.{self.loss}()")
        except AttributeError as e:
            raise (e, "Loss function you specified in config file doesn't exist!")

        self.current_step = 0
        self.current_epoch = 1
        self.best_loss = 1e7

        print(f"KaMi is ready and running on cuda")

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def __init_kami__(self):
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        self.optimizer = self.fetch_optimizer() or eval(
            f"torch.optim.{self.optimizer}"
        )(self.parameters(), lr=self.learning_rate, **self.optimizer_params)

        self.scheduler = self.fetch_scheduler() or eval(
            f"torch.optim.lr_scheduler.{self.scheduler}"
        )(self.optimizer, **self.scheduler_params)

        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        wandb.watch(self)

    def fit(self, train_loader, val_loader, resume=False):
        self.__init_kami__()
        if resume:
            self.load_checkpoint(
                os.path.join(self.output_dir, "last-checkpoint.pth"), resume=resume
            )

        for epoch in range(self.current_epoch, self.current_epoch + self.n_epochs):
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Current Learning rate {lr}")

            # Train one epoch
            train_loss, train_metrics = self.train_one_epoch(train_loader, epoch)
            wandb.log(
                {"train_loss": train_loss, **train_metrics}
            )

            # Save checkpoint
            self.save_checkpoint(
                epoch, os.path.join(self.output_dir or "", "last-checkpoint.pth")
            )

            val_loss, valid_metrics = self.validate(val_loader, epoch)
            wandb.log(
                {"valid_loss": val_loss, **valid_metrics}
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.eval()
                self.save_checkpoint(
                    epoch,
                    os.path.join(
                        self.output_dir,
                        f"best-checkpoint-{str(epoch).zfill(3)}.pth",
                    ),
                )
                for path in sorted(
                    glob(os.path.join(self.output_dir, f"best-checkpoint-*.pth"))
                )[:-1]:
                    os.remove(path)

        if self.send_model_file:
            shutil.copy(
                glob(os.path.join(self.output_dir, f"best-checkpoint-*.pth"))[-1],
                wandb.run.dir,
            )

    def train_one_step(self, data):
        _, loss, metrics = self.model_fn(data)
        with torch.set_grad_enabled(True):
            if self.fp16:
                with torch.cuda.amp.autocast():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()
        self.current_step += 1
        return loss, metrics

    def train_one_epoch(self, train_loader, epoch):
        self.train()
        summary_loss = AverageMeter()
        pbar = tqdm(
            train_loader, total=len(train_loader), desc=f"Training epoch {epoch}"
        )
        for index, data in enumerate(pbar):

            self.optimizer.zero_grad()

            loss, metrics = self.train_one_step(data)

            if index == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}

            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], train_loader.batch_size)
                monitor[f'train_{m_m}'] = metrics_meter[m_m].avg

            summary_loss.update(loss.detach().item(), train_loader.batch_size)
            pbar.set_postfix(loss=summary_loss.avg, stage="train", **monitor)
        pbar.close()
        return summary_loss.avg, monitor

    def validate(self, val_loader, epoch):
        self.eval()
        summary_loss = AverageMeter()
        pbar = tqdm(val_loader, total=len(val_loader), desc=f"Validate epoch {epoch}")
        for index, data in enumerate(pbar):
            with torch.no_grad():
                _, loss, metrics = self.model_fn(data)

            summary_loss.update(loss.detach().item(), val_loader.batch_size)
            if index == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}

            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], val_loader.batch_size)
                monitor[f'valid_{m_m}'] = metrics_meter[m_m].avg
            pbar.set_postfix(loss=summary_loss.avg, stage="valid", **monitor)
        pbar.close()
        
        return summary_loss.avg, monitor

    def predict_one_step(self, data):
        output, _, _ = self.model_fn(data)
        return output

    def predict(self, testloader, model_path=None):
        self.load_checkpoint(model_path)
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        if self.training:
            self.eval()

        pbar = tqdm(
            testloader,
            total=len(testloader),
            desc="Generating test dataset output!",
            leave=True,
        )
        for data in pbar:
            output = self.predict_one_step(data)
            output = self.process_output(output)
            yield output
            pbar.set_postfix(stage="test")
        pbar.close()

    def model_fn(self, data):
        d = {}
        if isinstance(data, list):
            d.update(
                {"inputs": data[0].to(self.device), "targets": data[1].to(self.device)}
            )
        else:
            for k, v in data.items():
                d.update(
                    {
                        k: v.to(self.device),
                    }
                )

        if self.fp16:
            with torch.cuda.amp.autocast():
                output, loss, metrics = self(**d)
        else:
            output, loss, metrics = self(**d)
        return output, loss, metrics

    def metrics(self, *args, **kwargs):
        return

    def loss_fn(self, *args, **kwargs):
        return

    def fetch_optimizer(self, *args, **kwargs):
        return

    def fetch_scheduler(self, *args, **kwargs):
        return

    @staticmethod
    def process_output(self, output):
        output = torch.argmax(output, dim=1).cpu().detach().numpy()
        return output

    @staticmethod
    def fetch_config(cfg):
        if isinstance(cfg, str):
            p = str(Path(cfg).absolute())
            assert os.path.exists(p), "Config file doesn't exist!"
            cfg = OmegaConf.load(p)
            print(OmegaConf.to_yaml(cfg))
            return OmegaConf.to_container(cfg)
        elif isinstance(cfg, dict):
            return cfg
        else:
            raise "Currently, KaMi support Dict or YAML config!"

    def save_checkpoint(self, epoch, path):
        self.eval()
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "fp16": self.fp16,
                "best_loss": self.best_loss,
                "epoch": epoch,
            },
            path,
        )

    def load_checkpoint(self, model_path, resume=False):
        if next(self.parameters()).device != self.device:
            self.to(self.device)
        if model_path is None:
            model_path = glob(os.path.join(self.output_dir, "best-checkpoint-*.pth"))[
                -1
            ]
        model_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.load_state_dict(model_dict["model_state_dict"])
        if resume:
            self.optimizer.load_state_dict(model_dict["optimizer_state_dict"])
            self.scheduler.load_state_dict(model_dict["scheduler_state_dict"])
            self.current_epoch = model_dict["epoch"]

    # Currently unavailable!!
    def generate_submission_file(self, test_loader, format=None):
        pass
