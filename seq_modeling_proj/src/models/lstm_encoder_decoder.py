import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(EncoderDecoder, self).__init__()

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=output_size,  # Decoder input size matches output size
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layer to map decoder output to final output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, y=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for the Encoder-Decoder model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            y (Tensor, optional): Ground truth tensor for teacher forcing. Shape: (batch_size, seq_len, output_size).
            teacher_forcing_ratio (float): Probability to use ground truth as next decoder input during training.

        Returns:
            Tensor: Predictions of shape (batch_size, seq_len, output_size).
        """
        # Encoder forward pass
        _, (hidden, cell) = self.encoder(x)

        # Initialize decoder input (zeros or ground truth)
        batch_size = x.size(0)
        seq_len = y.size(1) if y is not None else x.size(1)
        decoder_input = torch.zeros(batch_size, 1, self.fc.out_features, device=x.device)

        # Store predictions
        outputs = []

        # Decoder forward pass (one time step at a time)
        for t in range(seq_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            predicted_output = self.fc(decoder_output)  # Shape: (batch_size, 1, output_size)
            outputs.append(predicted_output)

            # Decide next input (teacher forcing vs. predicted output)
            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y[:, t, :].unsqueeze(1)  # Ground truth
            else:
                decoder_input = predicted_output  # Predicted output

        # Concatenate predictions along the time dimension
        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, seq_len, output_size)

        return outputs

# Example usage:
# model = EncoderDecoder(input_size=10, hidden_size=32, num_layers=2, output_size=5)
# x = torch.randn(16, 20, 10)  # Example input
# y = torch.randn(16, 20, 5)   # Example target (for teacher forcing)
# output = model(x, y, teacher_forcing_ratio=0.5)
# print(output.shape)  # Expected: (16, 20, 5)
