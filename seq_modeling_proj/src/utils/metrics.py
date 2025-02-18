import torch

def mse_loss(y_pred, y_true):
    """
    Compute Mean Squared Error (MSE) loss manually.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Scalar MSE loss.
    """
    return torch.mean((y_pred - y_true) ** 2)

# def rmse_loss(y_pred, y_true):
#     """
#     Compute Root Mean Squared Error (RMSE) loss.

#     Args:
#         y_pred (torch.Tensor): Predicted values.
#         y_true (torch.Tensor): Ground truth values.

#     Returns:
#         torch.Tensor: Scalar RMSE loss.
#     """
#     return torch.sqrt(mse_loss(y_pred, y_true))

def r2_score(y_pred, y_true):
    """
    Compute R-squared (coefficient of determination) score.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Scalar R² score.
    """
    ss_total = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    ss_residual = torch.sum((y_true - y_pred) ** 2, dim=0)
    r2_per_feature = 1 - (ss_residual / (ss_total + 1e-8))  # Avoid division by zero
    return torch.mean(r2_per_feature)  # Average across features
