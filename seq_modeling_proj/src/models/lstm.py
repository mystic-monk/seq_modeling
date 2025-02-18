# model.py
import lightning as pl
import torch
from torch import nn
import torch.nn.init as init

class LSTMRegressor(pl.LightningModule):
    """
    Standard PyTorch Lightning module:
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
        batch_size: int,  
        output_size: int = 1,
        **kwargs,
    ):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        # self.criterion = criterion
        # Convert string criterion to actual loss function
        if isinstance(criterion, str):
            self.criterion = getattr(nn, criterion)()  # Example: "MSELoss" -> nn.MSELoss()
        else:
            self.criterion = criterion 

        self.save_hyperparameters(ignore=["criterion"])
        self.learning_rate = learning_rate


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
        # Xavier initialization
        self._initialize_weights()

        # Stateful hidden and cell states
        self.h0 = None
        self.c0 = None

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0)

                
    def reset_states(self):
        """
        Reset hidden and cell states to zeros.
        """

        self.h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)

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
        # loss.backward(retain_graph=True)
        loss.backward()


    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        Returns:
            dict: Optimizer and scheduler configuration.
        """
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # optimizer = torch.optim.SGD(self.parameters(), 
        #                             lr=self.learning_rate, 
        #                             weight_decay=0.0005, 
        #                             dampening=0.0, 
        #                             momentum=0.9, 
        #                             nesterov=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5)

        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            }
