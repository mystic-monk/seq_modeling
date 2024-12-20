# model.py
import lightning as L
import torch
from torch import nn

from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score



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
        output_size: int =1,
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

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )

        # fully connected layer/ Dense Layer
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        """
        Forward pass of the LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
        Returns:
            torch.Tensor: Predicted output of shape (batch_size, output_size).
        """
        lstm_out, _ = self.lstm(x)

        y_pred = self.fc(lstm_out[:, -1])

        return y_pred

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        Returns:
            dict: Optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
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
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
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
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate additional metrics
        val_mae = self.mae(y_hat, y)
        val_mse = self.mse(y_hat, y)
        val_rmse = torch.sqrt(val_mse)  
        val_r2 = self.r2(y_hat, y)

        # log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_mae", val_mae, on_step=False, on_epoch=True)
        self.log("val_mse", val_mse, on_step=False, on_epoch=True)
        self.log("val_rmse", val_rmse, on_step=False, on_epoch=True)
        self.log("val_r2", val_r2, on_step=False, on_epoch=True)

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
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        # Calculate additional metrics
        test_mae = self.mae(y_hat, y)
        test_mse = self.mse(y_hat, y)
        test_rmse = torch.sqrt(test_mse) 
        test_r2 = self.r2(y_hat, y)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mae", test_mae, on_step=False, on_epoch=True)
        self.log("test_mse", test_mse, on_step=False, on_epoch=True)
        self.log("test_rmse", test_rmse, on_step=False, on_epoch=True)
        self.log("test_r2", test_r2, on_step=False, on_epoch=True)

        return loss

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
        print("Running predict_step.")
        x, _ = batch
        y_pred = self.forward(x)
        
        return y_pred
