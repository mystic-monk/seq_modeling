# model.py
import lightning as pl
import torch
from torch import nn
# from utils import logger
# from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, MeanAbsolutePercentageError
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
        # debug: bool = False,
        **kwargs,
    ):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.criterion = criterion
        self.save_hyperparameters(ignore=["criterion"])
        self.learning_rate = learning_rate
        # self.debug = debug  

        # Initialize metrics
        # self.mae = MeanAbsoluteError()
        # self.mse = MeanSquaredError()
        # self.r2 = R2Score()
        # self.mape = MeanAbsolutePercentageError()

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

        # self.train_epoch_y = []
        # self.train_epoch_y_hat = []
        # self.test_epoch_y = []
        # self.test_epoch_y_hat = []
        # self.val_epoch_y = []
        # self.val_epoch_y_hat = []

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0)

                
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
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        optimizer = torch.optim.SGD(self.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=0.0005, 
                                    dampening=0.0, 
                                    momentum=0.9, 
                                    nesterov=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(

        #     optimizer, T_0=10, T_mult=2, verbose=False
        #     )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            }

    # def training_step(self, batch, batch_idx):
    #     """
    #     Training step logic.
    #     Args:
    #         batch (tuple): Input batch containing (x, y).
    #         batch_idx (int): Batch index.
    #     Returns:
    #         torch.Tensor: Training loss.
    #     """
    #     if self.debug:
    #         logger.debug("Training step: Start.")
        
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     if self.debug:
    #         logger.debug(f"Batch : {batch_idx} X shape: {x.shape}, Y shape: {y.shape}")


    #     # Compute the loss
    #     loss = self.criterion(y_hat, y)

    #     # save the values for epoch end caculations
    #     self.train_epoch_y.append(y.detach().cpu())   
    #     self.train_epoch_y_hat.append(y_hat.detach().cpu())

    #     # Log metrics
    #     self.log("batch_loss", loss.item(), on_step=True, on_epoch=False)
    #     self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)


    #     if self.debug:
    #         logger.debug(f"Training loss: {loss.item()}")
    #     # Return only the loss
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     """
    #     Validation step logic.
    #     Args:
    #         batch (tuple): Input batch containing (x, y).
    #         batch_idx (int): Batch index.
    #     Returns:
    #         torch.Tensor: Validation loss.
    #     """
    #     if self.debug:
    #         logger.debug("Validation step: Start.")
    #     x, y = batch

    #     y_hat = self.forward(x)
        
    #     # Calculate the loss
    #     loss = self.criterion(y_hat, y)
        
    #     # save the values for epoch end caculations
    #     self.val_epoch_y.append(y.detach().cpu())   
    #     self.val_epoch_y_hat.append(y_hat.detach().cpu())


    #     # # Log metrics
    #     self.log("val_loss", loss.item(), on_step=False, on_epoch=True)


    #     if self.debug:  
    #         logger.debug(f"Validation loss: {loss.item()}")

    #     return loss


    # def test_step(self, batch, batch_idx):
    #     """
    #     Test step logic.
    #     Args:
    #         batch (tuple): Input batch containing (x, y).
    #         batch_idx (int): Batch index.
    #     Returns:
    #         torch.Tensor: Test loss.
    #     """
    #     if self.debug:
    #         logger.debug("Test step: Start.")
    #     x, y = batch

    #     if self.debug:
    #         logger.debug(f"Batch : {batch_idx} X shape: {x.shape}, Y shape: {y.shape}")

    #     y_hat = self.forward(x)
    #     print(f"in test step {batch_idx}")
    #     print(f"Y Hat {y_hat}")
    #     if self.debug:
    #         logger.debug(f"y_hat shape: {y_hat.shape}")

    #     # Calculate the loss
    #     loss = self.criterion(y_hat, y)

    #     # save the values for epoch end caculations
    #     self.test_epoch_y.append(y.detach().cpu())   
    #     self.test_epoch_y_hat.append(y_hat.detach().cpu())

    #     # Log metrics
    #     self.log("test_loss", loss.item(), on_step=False, on_epoch=True)


    #     if self.debug:

    #         logger.debug(f"Test loss: {loss.item()}")
    #     return loss


    # def on_train_epoch_start(self):
    #     """
    #     Reset states at the start of each training epoch.
    #     """
    #     self.reset_states(batch_size=self.hparams.batch_size)

    # def on_train_epoch_end(self):
    #     """
    #     Compute and log training metrics at the end of each epoch.
    #     """
    #     self.train_epoch_y_hat.clear()
    #     self.train_epoch_y.clear()
        

    # def on_validation_epoch_start(self):
    #     """
    #     Reset states at the start of each validation epoch.
    #     """
    #     self.reset_states(batch_size=self.hparams.batch_size)
        
    # def on_validation_epoch_end(self):
        

    #     self.val_epoch_y.clear()
    #     self.val_epoch_y_hat.clear()


    # def on_test_epoch_start(self):
    #     """
    #     Reset states at the start of each test epoch.
    #     """

    #     self.reset_states(batch_size=self.hparams.batch_size)


    # def on_test_epoch_end(self):
    #     print("Test Epoch End")
    #     print(f"Test Y shape {len(self.test_epoch_y)}")
    #     print(f"Test Y Hat shape {len(self.test_epoch_y_hat)}")
    #     print(f"Test Y {self.test_epoch_y}")
    #     print(f"Test Y Hat {self.test_epoch_y_hat}")
    #     testm_y_all = torch.cat(self.test_epoch_y, dim=0) 
    #     testm_y_hat_all = torch.cat(self.test_epoch_y_hat, dim=0)
    #     print(f"Test Y shape {testm_y_all.shape}")
    #     print(f"Test Y Hat shape {testm_y_hat_all.shape}")

    #     test_r2_adj = self.compute_adjusted_r2(testm_y_all, testm_y_hat_all, self.n_features)
    
    #     # Log adjusted R^2
    #     self.log("test_r2_adj", test_r2_adj.item(), on_step=False, on_epoch=True)

    #     r2B = self.compute_r2(testm_y_all, testm_y_hat_all)

    #     test_r2 = self.r2(testm_y_hat_all, testm_y_all)
    #     test_mse = self.mse(testm_y_hat_all, testm_y_all)
    #     test_rmse = torch.sqrt(test_mse)
    #     test_mae = self.mae(testm_y_hat_all, testm_y_all)
        
    #     test_residuals, test_residuals_mean, self.test_residuals_sd =self.compute_residuals(testm_y_all, testm_y_hat_all)

    #     # # MAPE is unreliable when true values (𝑦) are close to zero.
    #     test_mape = self.mape(testm_y_hat_all, testm_y_all)
                
    #     print(f"R2 Score  {test_r2}")
    #     print(f"R2B Score {r2B}")
        
    #     self.log("test_r2", test_r2, on_step=False, on_epoch=True)
    #     self.log("test_mse", test_mse.item(), on_step=False, on_epoch=True)
    #     self.log("test_rmse", test_rmse.item(), on_step=False, on_epoch=True)
    #     self.log("test_mae", test_mae.item(), on_step=False, on_epoch=True)
    #     self.log("test_mape", test_mape.item(), on_step=False, on_epoch=True)
    #     self.log("test_residuals_mean", test_residuals_mean, on_step=False, on_epoch=True)

    #     self.test_epoch_y.clear()
    #     self.test_epoch_y_hat.clear()


    # def compute_r2(self, y_true, y_pred):
    #     ss_res = torch.sum((y_true - y_pred) ** 2)
    #     ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    #     print(f"SS Res {ss_res}")
    #     print(f"SS Tot {ss_tot}")
    #     return 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(float('nan'))


    # def compute_residuals(self, y_true, y_pred):
    #     residuals = y_true - y_pred
    #     return residuals, torch.mean(residuals), torch.std(residuals)


    # def compute_adjusted_r2(self, y_true, y_pred, num_features):
    #     """
    #     Compute Adjusted R^2 Score.
    #     Args:
    #         y_true (torch.Tensor): Ground truth values.
    #         y_pred (torch.Tensor): Predicted values.
    #         num_features (int): Number of predictors (features).
    #     Returns:
    #         torch.Tensor: Adjusted R^2 score.
    #     """
    #     ss_res = torch.sum((y_true - y_pred) ** 2)
    #     ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    #     n = y_true.size(0)  # Number of observations
    #     r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(float('nan'))
        
    #     if n <= num_features + 1:  # Avoid invalid computation
    #         return r2  # Return unadjusted R^2 if n is too small
        
    #     r2_adj = 1 - ((1 - r2) * (n - 1) / (n - num_features - 1))
    #     return r2_adj


    # def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
    #     """
    #     Predict step logic for inference.
    #     Args:
    #         batch (tuple): Input batch containing (x, y).
    #         batch_idx (int): Batch index.
    #         dataloader_idx (int): Dataloader index."""
    #     x, _ = batch
    #     y_pred, _ = self.forward(x)
    #     return y_pred
