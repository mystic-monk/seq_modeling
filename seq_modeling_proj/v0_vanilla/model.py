# model.py
import lightning as L
import torch
from torch import nn

from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, MeanAbsolutePercentageError

class LSTMRegressor(L.LightningModule):
    """
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    LSTM-based regressor implemented as a PyTorch Lightning Module.
    Suitable for sequence modeling tasks with numerical outputs.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        criterion: nn.Module,
        batch_size: int,  # Add batch_size as a hyperparameter
        output_size: int = 14,  # Update output_size to 14
        **kwargs,
    ):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.save_hyperparameters(ignore=["criterion"])
        self.learning_rate = learning_rate

        # Initialize metrics
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        self.mape = MeanAbsolutePercentageError()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Stateful hidden and cell states
        self.h0 = None
        self.c0 = None

    def reset_states(self, batch_size: int):
        """
        Reset hidden and cell states to zeros.
        """
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

    def forward(self, x):
        """
        Forward pass with stateful hidden and cell states.
        """
        if self.h0 is None or self.c0 is None:
            self.reset_states(x.size(0))
        lstm_out, (h0, c0) = self.lstm(x, (self.h0.to(x.device), self.c0.to(x.device)))
        (self.h0, self.c0) = (h0.detach(), c0.detach())
        y_pred = self.fc(lstm_out[:, -1])
        return y_pred



    def backward(self, loss):
        """
        Backward pass with truncated backpropagation through time (TBPTT).
        """
        loss.backward(retain_graph=True)


    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        Returns:
            dict: Optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        """
        Training step logic.
        Args:
            batch (tuple): Input batch containing (x, y).
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Training loss.
        """
        x, y = batch

        y = y[:, -1]
        y = y.squeeze(-1)  # Squeeze the extra dimension from the target tensor
        
        y_hat = self.forward(x)

        # Compute the loss
        loss = self.criterion(y_hat, y)

        # Calculate residuals
        residuals = y - y_hat

        # Log mean of residuals
        self.log("train_residuals_mean", residuals.mean(), on_step=True, on_epoch=True)

        # Log the loss
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True)
        
        # print(f"Batch {batch_idx}, Loss: {loss.item()}", end="\r")
        # Return only the loss
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step logic.
        Args:
            batch (tuple): Input batch containing (x, y).
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Validation loss.
        """
        x, y = batch

        y = y[:, -1]
        y = y.squeeze(-1)  # Squeeze the extra dimension from the target tensor

        # y_hat, (ht, ct) = self.forward(x, h0, c0)
        y_hat = self.forward(x)

        # Ensure tensors are contiguous
        y = y.contiguous()
        y_hat = y_hat.contiguous()
        
        # Calculate the loss
        loss = self.criterion(y_hat, y)
        
        # Calculate residuals
        residuals = y - y_hat

        # Log mean of residuals
        self.log("val_residuals_mean", residuals.mean(), on_step=False, on_epoch=True)

        # Calculate additional metrics
        val_mae = self.mae(y_hat, y)    
        val_mse = self.mse(y_hat, y)
        val_rmse = torch.sqrt(val_mse)  
        val_r2 = self.r2(y_hat, y) if y.numel() > 1 else torch.tensor(float('nan'))
        val_mape = self.mape(y_hat, y)

        # log metrics
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True)
        self.log("val_mae", val_mae.item(), on_step=True, on_epoch=True)
        self.log("val_mse", val_mse.item(), on_step=True, on_epoch=True)
        self.log("val_rmse", val_rmse.item(), on_step=True, on_epoch=True)
        self.log("val_r2", val_r2.item(), on_step=True, on_epoch=True)
        self.log("val_mape", val_mape.item(), on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step logic.
        Args:
            batch (tuple): Input batch containing (x, y).
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Test loss.
        """
        x, y = batch
        y = y[:, -1]
        y = y.squeeze(-1)  # Squeeze the extra dimension from the target tensor
        # y_hat, (ht, ct) = self.forward(x)
        y_hat = self.forward(x)

        # Ensure tensors are contiguous
        y = y.contiguous()
        y_hat = y_hat.contiguous()

        # Calculate the loss
        loss = self.criterion(y_hat, y)

        # Calculate residuals
        residuals = y - y_hat

        # Log mean of residuals
        self.log("test_residuals_mean", residuals.mean(), on_step=False, on_epoch=True)

        # Calculate additional metrics
        test_mae = self.mae(y_hat, y)
        test_mse = self.mse(y_hat, y)
        test_rmse = torch.sqrt(test_mse)
        test_r2 = self.r2(y_hat, y) if y.numel() > 1 else torch.tensor(float('nan'))
        test_mape = self.mape(y_hat, y)

        # Log metrics
        self.log("test_loss", loss.item(), on_step=False, on_epoch=True)
        self.log("test_mae", test_mae.item(), on_step=False, on_epoch=True)
        self.log("test_mse", test_mse.item(), on_step=False, on_epoch=True)
        self.log("test_rmse", test_rmse.item(), on_step=False, on_epoch=True)
        self.log("test_r2", test_r2.item(), on_step=False, on_epoch=True)
        self.log("test_mape", test_mape.item(), on_step=False, on_epoch=True)

        return loss


    def on_train_epoch_start(self):
        """
        Reset states at the start of each training epoch.
        """
        self.reset_states(batch_size=self.hparams.batch_size)


    def on_validation_epoch_start(self):
        """
        Reset states at the start of each validation epoch.
        """
        self.reset_states(batch_size=self.hparams.batch_size)

    def on_test_epoch_start(self):
        """
        Reset states at the start of each test epoch.
        """
        self.reset_states(batch_size=self.hparams.batch_size)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        """
        Predict step logic for inference.
        Args:
            batch (torch.Tensor): Input batch (x).
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index (for multi-dataloader scenarios).
        Returns:
            torch.Tensor: Predicted outputs.
        """
        x, _ = batch
        y_pred, _ = self.forward(x)
        return y_pred
