from .lstm_base import LSTM
import torch

class LSTMClassifier(torch.nn.Module):
    def __init__(
        self,
        n_classes,
        hidden_size,
        vocab_size,
        embedding_dim=512,
        PAD=None,
        dropout_p=0.3
    ):
        """
        Simple text classifier, which extract features with lstm and uses one hidden layer
        :param n_classes: number of classes for classification
        :param hidden_size: dim of LSTM`s hidden layer
        :param vocab_size: size of vocabulary, which encodes texts
        :param embedding_dim: dim of embeddings to feed to lstm
        :param PAD: padding token id, necessary for better averaging of inputs, if left None
            result is all outputs average
        :param dropout_p: parameter for dropout in hidden layer
        :return: logits, torch.Tensor of size (batch_size, n_classes)
        """
        super().__init__()

        self.embedding = torch.nn.Embedding(
                    vocab_size,
                    embedding_dim=embedding_dim
                )
        self.lstm = LSTM(
            embedding_dim,
            hidden_size
        )

        self.linear1 = torch.nn.Linear(hidden_size, 2 * hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.linear2 = torch.nn.Linear(2 * hidden_size, n_classes)

        self.PAD = PAD

    def forward(self, x):
        mask = (x != self.PAD)
        lens = mask.sum(dim=1)

        x = self.embedding(x)
        h = self.lstm(x)

        # averaging over all outputs excluding outputs, which correspond to [PAD] tokens
        output_padding_aware = h * mask.unsqueeze(2)
        output_summed = output_padding_aware.sum(dim=1)
        output_averaged = output_summed / lens.unsqueeze(1)

        lin1 = self.dropout(self.relu(self.linear1(output_averaged)))
        lin2 = self.linear2(lin1)

        return lin2