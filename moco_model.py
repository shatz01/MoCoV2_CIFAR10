import pytorch_lightning as pl
import lightly
import torch.nn as nn
import torch
import numpy as np
import copy

class MocoModel(pl.LightningModule):
    def __init__(self, memory_bank_size, moco_max_epochs, downstream_max_epochs=0, dataloader_train_classifier=None, dataloader_test=None, downstream_test_every=0):
        super().__init__()

        self.moco_max_epochs = moco_max_epochs
        self.downstream_max_epochs = downstream_max_epochs
        self.dataloader_train_classifier = dataloader_train_classifier
        self.dataloader_test = dataloader_test
        self.downstream_test_every = downstream_test_every

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=8)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(backbone, num_ftrs=512, m=0.99, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    # freeze backbone and train a few linear layers to see progress
    def test_downstream_training(self):
        # copy moco and make classifier
        classifier = Classifier(copy.deepcopy(self.resnet_moco), max_epochs = self.downstream_max_epochs)
        trainer = pl.Trainer(max_epochs = self.downstream_max_epochs, gpus=1, logger=None)
        trainer.fit(
                classifier,
                self.dataloader_train_classifier,
                self.dataloader_test
                )
        train_losses = classifier.train_losses[1:]
        val_accs = classifier.val_accs[1:]
        print(train_losses)
        print(val_accs)
        print('-----')
        min_train_loss = np.min(train_losses)
        max_val_acc = np.max(val_accs)
        self.log('DOWNSTREAM_min_train_loss', min_train_loss)
        self.log('DOWNSTREAM_max_val_acc', max_val_acc)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_moco(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()
        if self.current_epoch%self.downstream_test_every == 0:
            print('... training downstream classifier...')
            self.test_downstream_training()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.moco_max_epochs)
        return [optim], [scheduler]


class Classifier(pl.LightningModule):
    def __init__(self, model, max_epochs):
        super().__init__()
        # create a moco based on ResNet
        self.resnet_moco = model
        self.max_epochs = max_epochs
        self.epoch_train_losses = []
        self.epoch_val_accs = []
        self.train_losses = []
        self.val_accs = []

        # freeze the layers of moco
        for p in self.resnet_moco.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
                )

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.resnet_moco.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        self.epoch_train_losses.append(loss.cpu().detach())
        return loss
    
    # TODO: logging histogram doesnt work when using nn.Sequenial for model fc
    # def training_epoch_end(self, outputs):
    #     self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        self.accuracy(y_hat, y)
        val_acc = self.accuracy.compute()
        self.log('val_acc', val_acc,
                 on_epoch=True, prog_bar=True)
        self.epoch_val_accs.append(val_acc.cpu().detach())

    def validation_epoch_end(self, outputs):
        self.train_losses.append(np.mean(self.epoch_train_losses))
        self.val_accs.append(np.mean(self.epoch_val_accs))

    def configure_optimizers(self):
        # IDK why but lr=3 works good when we use 3 layers in fc. They had an lr=30. when it was just 1 layer lol
        optim = torch.optim.SGD(self.fc.parameters(), lr=3.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
