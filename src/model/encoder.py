import torch
from torch.nn import Module, LSTM, Linear

class Encoder(Module):
    """
    Encoder module for the LSTM-based model.

    Args:
        input_size (int): Input size of the encoder.
        hidden_size (int): Hidden size of the LSTM layer.
        seq_size (int): Sequence size of the input batch.
        batch_size (int): Batch size of the input batch.

    Examples:
        # Create an Encoder instance
        >>> encoder = Encoder(input_size=128, hidden_size=256, seq_size=10, batch_size=32)

        # Forward pass through the encoder
        >>> output, hidden, c = encoder.forward(batch)
    """
    def __init__(self, input_size, hidden_size, seq_size, batch_size) -> None:
        """
        Initialize the Encoder module.

        Args:
            input_size (int): Input size of the encoder.
            hidden_size (int): Hidden size of the LSTM layer.
            seq_size (int): Sequence size of the input batch.
            batch_size (int): Batch size of the input batch.

        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.blstm = LSTM(input_size, hidden_size, bidirectional=True)  # Bidirectional LSTM layer
        self.linear = Linear(input_size, 256)  # Linear layer for optional output transformation

    def forward(self, batch, hidden=None, linear=False):
        """
        Forward pass of the Encoder module.

        Args:
            batch (torch.Tensor): Input batch of shape (seq_size, batch_size, input_size).
            hidden (tuple, optional): Initial hidden state for LSTM layer. Defaults to None.
            linear (bool, optional): Whether to apply linear layer for output transformation. Defaults to False.

        Returns:
            tuple: Tuple containing:
                - output (torch.Tensor): Output tensor from LSTM layer.
                - hidden (torch.Tensor): Hidden state tensor from LSTM layer.
                - c (torch.Tensor): Cell state tensor from LSTM layer.
        """

        # Preprocess input batch if needed
        if batch.shape != (self.seq_size, self.batch_size, self.input_size):
            preprocess = torch.transpose(torch.transpose(torch.flatten(batch, -2, -1), -2, -1), 0, 1)
        else:
            preprocess = batch

        # Perform forward pass through BLSTM layer
        if hidden:
            output, (hidden_state, cell_state) = self.blstm(preprocess, hidden)
        else:
            output, (hidden_state, cell_state) = self.blstm(preprocess)

        # Apply linear layer if specified
        if linear:
            output = self.linear(output)
            hidden = self.linear(hidden.transpose(0, 1).flatten(-2, -1))

        return output, hidden_state, cell_state
