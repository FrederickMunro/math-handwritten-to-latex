import torch
from torch.nn import Module

class HME2LaTeX(Module):
    """
    HME2LaTeX model for converting handwritten mathematical expressions to LaTeX.

    Args:
        cnn (torch.nn.Module): CNN model for feature extraction.
        encoder (torch.nn.Module): Encoder model for generating initial hidden states and cell states.
        decoder (torch.nn.Module): Decoder model for generating LaTeX expressions.
        seq_size (int): Size of the sequence to be generated.
        batch_size (int): Batch size for input images and output LaTeX sequences.
        language_size (int): Size of the LaTeX language vocabulary.
        device (torch.device): Device on which the model is run.

    Returns:
        torch.Tensor: Output tensor containing probabilities of the generated LaTeX sequence.
    """
    def __init__(self, cnn, encoder, decoder, seq_size, batch_size, language_size, device):
        super(HME2LaTeX, self).__init__()
        self.cnn = cnn
        self.encoder = encoder
        self.decoder = decoder
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.language_size = language_size
        self.device = device

    def forward(self, images, labels):
        """
        Perform forward pass through the HME2LaTeX model.

        Args:
            images (torch.Tensor): Input images to be processed by the CNN model.
            labels (torch.Tensor or None): Labels representing the target LaTeX sequence during training,
                or None during testing.

        Returns:
            torch.Tensor: Output tensor containing probabilities of the generated LaTeX sequence.
        """

        # CNN forward pass
        cnn_output = self.cnn(images)

        # Encoder forward pass
        encoder_output, encoder_hidden, encoder_cell = self.encoder(cnn_output)

        # Concatenates the forward and backward hidden states and expands them into a new dimension
        forward_hidden_state = encoder_hidden[0]
        backward_hidden_state = encoder_hidden[1]
        init_hidden = torch.cat((forward_hidden_state, backward_hidden_state), -1).unsqueeze(0).to(self.device)

        # Concatenates the forward and backward cell states and expands them into a new dimension
        forward_cell_state = encoder_cell[0]
        backward_cell_state = encoder_cell[1]
        init_cell = torch.cat((forward_cell_state, backward_cell_state), -1).unsqueeze(0).to(self.device)
        

        # Initialize probability tensor
        probs = torch.zeros((self.seq_size, self.batch_size, self.language_size)).to(self.device)

        out, hidden, prob = self.decoder(labels.unsqueeze(0), (init_hidden, init_cell), encoder_output)
        probs[0, :, :] += prob[0]

        return probs
