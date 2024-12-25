import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

class HybridModel(LightningModule):  # Updated base class
    def __init__(self, n_features, hidden_size, num_layers, dropout, learning_rate, criterion, output_size, attention_heads, attention_layers):
        super(HybridModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.output_size = output_size
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers

        # Define LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Define Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)

        # Apply attention
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Pooling over time dimension (mean pooling)
        pooled_attention = attention_out.mean(dim=1)

        # Fully connected layer
        output = self.fc(pooled_attention)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
