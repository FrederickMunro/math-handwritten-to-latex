import torch
from torch.nn import Module, Linear, LSTM, Softmax

class Decoder(Module):
    """
    Decoder module for a sequence-to-sequence model with attention mechanism.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state in LSTM.
        output_size (int): Size of output features.
        num_features (int): Number of features in input sequence.
        batch_size (int): Batch size of input sequence.
        device (torch.device): Device on which the model is run.

    Attributes:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state in LSTM.
        output_size (int): Size of output features.
        num_features (int): Number of features in input sequence.
        batch_size (int): Batch size of input sequence.
        device (torch.device): Device on which the model is run.
        lstm (torch.nn.LSTM): LSTM layer for sequential processing of input.
        wout (torch.nn.Linear): Linear layer for output.
        wf (torch.nn.Linear): Linear layer for attention weights on encoder output.
        wh (torch.nn.Linear): Linear layer for attention weights on hidden state.
        wc (torch.nn.Linear): Linear layer for combined attention weights.
        softmax_out (torch.nn.Softmax): Softmax activation for output probabilities.
        softmax_alpha (torch.nn.Softmax): Softmax activation for attention weights on hidden state.

    Examples:
        # Create a Decoder instance
        >>> decoder = Decoder(input_size=256, hidden_size=512, output_size=128,
                          num_features=10, batch_size=32, device=torch.device('cuda'))

        # Forward pass through the decoder
        >>> output, hidden, probability = decoder(input, hidden_state, encoder_output)
    """
    def __init__(self, input_size, hidden_size, output_size, num_features, batch_size, device):
        """
        Initialize the Decoder module.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Size of hidden state in LSTM.
            output_size (int): Size of output features.
            num_features (int): Number of features in input sequence.
            batch_size (int): Batch size of input sequence.
            device (torch.device): Device on which the model is run.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_features = num_features
        self.batch_size = batch_size
        self.device = device
        self.lstm = LSTM(self.input_size, self.hidden_size) # LSTM layer for sequential processing of input
        self.wout = Linear(self.hidden_size, self.output_size, bias=False) # Linear layer for output
        self.wf = Linear(self.hidden_size, 1, bias=False) # Linear layer for attention weights on encoder output
        self.wh = Linear(self.hidden_size, 1, bias=False) # Linear layer for attention weights on hidden state
        self.wc = Linear(2 * self.hidden_size, self.hidden_size, bias=False) # Linear layer for combined attention weights
        self.softmax_out = Softmax(dim=-1) # Softmax activation for output probabilities
        self.softmax_alpha = Softmax(dim=0) # Softmax activation for attention weights on hidden state

    def forward(self, input, hidden, encoder_output):
        """
        Perform forward pass of the decoder module.

        Args:
            input (torch.Tensor): Input tensor of shape (sequence_length, batch_size, input_size).
            hidden (tuple): Tuple of hidden state and cell state of the LSTM, each of shape (num_layers, batch_size, hidden_size).
            encoder_output (torch.Tensor): Encoder output tensor of shape (sequence_length, batch_size, hidden_size).

        Returns:
            tuple: Tuple containing:
                - output (torch.Tensor): Output tensor of shape (sequence_length, batch_size, output_size).
                - hidden (tuple): Updated hidden state and cell state of the LSTM, each of shape (num_layers, batch_size, hidden_size).
                - probability (torch.Tensor): Probability tensor of shape (sequence_length, batch_size, output_size).
        """
        hidden_states, cell_states = hidden

        # Calculate attention coefficients (alphas)
        alphas = self.softmax_alpha(
            torch.tanh(
                self.wh(hidden_states.expand(self.num_features, self.batch_size, self.hidden_size))
                + self.wf(encoder_output)
            )
        ).to(self.device)

        c = torch.sum(alphas * encoder_output, 0, keepdim=True).to(self.device) # Compute context vector (c)
        new_h = torch.cat((hidden_states, c), -1).to(self.device)
        new_h = torch.tanh(self.wc(new_h)).to(self.device)
        output, (hidden, cell_states) = self.lstm(input, (new_h, cell_states)) # LSTM forward pass
        probability = self.wout(output) # Compute output probabilities
        return output, (new_h, cell_states), probability