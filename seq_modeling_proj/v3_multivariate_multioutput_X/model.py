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

    def forward(self, x, h0=None, c0=None):
        """
        Forward pass of the LSTM model with hidden and cell states.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            h0 (torch.Tensor): Initial hidden state (optional).
            c0 (torch.Tensor): Initial cell state (optional).
        Returns:
            torch.Tensor: Predicted output of shape (batch_size, output_size).
            tuple: Updated hidden and cell states (ht, ct).
        """
        if h0 is None or c0 is None:
            if x.dim() == 2:  # Unbatched input
                h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
            else:  # Batched input
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_out, (ht, ct) = self.lstm(x.unsqueeze(0) if x.dim() == 2 else x, (h0, c0))
        y_pred = self.fc(lstm_out[:, -1])
        return y_pred, (ht, ct)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        Returns:
            dict: Optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx, h0=None, c0=None):
        """
        Training step logic.
        Args:
            batch (tuple): Input batch containing (x, y).
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Training loss.
        """
        x, y = batch
        y = y.squeeze(-1)  # Squeeze the extra dimension from the target tensor
        
        # Initialize hidden and cell states if not provided
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        y_hat, (ht, ct) = self.forward(x, h0, c0)  # Forward pass with states

        # Compute the loss
        loss = self.criterion(y_hat, y)

        # Calculate residuals
        residuals = y - y_hat

        # Log mean of residuals
        self.log("train_residuals_mean", residuals.mean(), on_step=True, on_epoch=True)

        # Return the updated states for the next batch (do not store in self directly)
        self.h0, self.c0 = ht, ct  # Update internal states
        
        # Log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        
        # Return only the loss
        return loss

    def validation_step(self, batch, batch_idx, h0=None, c0=None):
        """
        Validation step logic.
        Args:
            batch (tuple): Input batch containing (x, y).
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Validation loss.
        """
        x, y = batch
        y = y.squeeze(-1)  # Squeeze the extra dimension from the target tensor
        y_hat, (ht, ct) = self.forward(x, h0, c0)
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
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_mae", val_mae, on_step=False, on_epoch=True)
        self.log("val_mse", val_mse, on_step=False, on_epoch=True)
        self.log("val_rmse", val_rmse, on_step=False, on_epoch=True)
        self.log("val_r2", val_r2, on_step=False, on_epoch=True)
        self.log("val_mape", val_mape, on_step=False, on_epoch=True)

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
        y = y.squeeze(-1)  # Squeeze the extra dimension from the target tensor
        y_hat, (ht, ct) = self.forward(x)
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
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mae", test_mae, on_step=False, on_epoch=True)
        self.log("test_mse", test_mse, on_step=False, on_epoch=True)
        self.log("test_rmse", test_rmse, on_step=False, on_epoch=True)
        self.log("test_r2", test_r2, on_step=False, on_epoch=True)
        self.log("test_mape", test_mape, on_step=False, on_epoch=True)

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
        y_pred, _ = self.forward(x)
        return y_pred
