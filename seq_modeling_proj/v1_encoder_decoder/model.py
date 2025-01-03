# model.py
import lightning as L
import torch
from torch import nn

from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, MeanAbsolutePercentageError


class Encoder(nn.Module):
    """
    LSTM Encoder for sequence modeling.
    """
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, n_features) - Input sequence.
        Returns:
            outputs: (batch_size, seq_len, hidden_size) - LSTM outputs for all time steps.
            hidden, cell: (num_layers, batch_size, hidden_size) - Final hidden and cell states.
        """
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class Decoder(nn.Module):
    """
    LSTM Decoder for sequence-to-sequence tasks without attention.
    """
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size + output_size,  # Encoder hidden state + previous output
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        """
        Args:
            x: (batch_size, 1, output_size) - Previous output (initially a start token or zeros).
            hidden, cell: (num_layers, batch_size, hidden_size) - Decoder's hidden and cell states.
        Returns:
            output: (batch_size, output_size) - Predicted output for the current time step.
            hidden, cell: Updated hidden and cell states.
        """
        lstm_input = torch.cat((x, hidden[-1].unsqueeze(1)), dim=2)  # Concatenate previous output and hidden state
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.fc(output.squeeze(1))  # Apply fully connected layer to LSTM output
        return output, hidden, cell
    

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
        self.save_hyperparameters(ignore=["criterion"])
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_size = output_size

        # Initialize encoder and decoder
        self.encoder = Encoder(n_features, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, output_size, num_layers, dropout)

        # Initialize metrics
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        self.mape = MeanAbsolutePercentageError()

    def forward(self, x):
        """
        Forward pass of the encoder-decoder model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
        Returns:
            torch.Tensor: Predicted output of shape (batch_size, output_size, output_size).
        """
        encoder_outputs, hidden, cell = self.encoder(x)  # Get hidden and cell states from encoder
        decoder_input = torch.zeros(x.size(0), 1, self.output_size).to(x.device)  # Initial input to the decoder

        outputs = []
        for _ in range(self.output_size):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)  # Update hidden and cell states
            outputs.append(output.unsqueeze(1))  # Add an extra dimension for concatenation
            decoder_input = output.unsqueeze(1)  # Use the current output as the next input

        outputs = torch.cat(outputs, dim=1)  # Concatenate along the sequence dimension
        return outputs

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
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True)
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
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True)
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
        self.log("test_loss", loss.item(), on_step=False, on_epoch=True)
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
        x, _ = batch
        y_pred = self.forward(x)
        return y_pred
