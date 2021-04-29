import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        LSTM with input, forget and output gates
        :param input_size: dim of input vectors
        :param hidden_size: dim of hidden layer
        :return: all outputs in a matrix size (hidden_size, sequence_len)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size


        self.W_ii = torch.nn.Linear(input_size, hidden_size)
        self.W_hi = torch.nn.Linear(hidden_size, hidden_size)

        self.W_if = torch.nn.Linear(input_size, hidden_size)
        self.W_hf = torch.nn.Linear(hidden_size, hidden_size)

        self.W_ig = torch.nn.Linear(input_size, hidden_size)
        self.W_hg = torch.nn.Linear(hidden_size, hidden_size)

        self.W_io = torch.nn.Linear(input_size, hidden_size)
        self.W_ho = torch.nn.Linear(hidden_size, hidden_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        batch_size, sequence_len, _ = x.shape

        outputs = []

        h = torch.zeros((batch_size, self.hidden_size), device=x.device)
        c = torch.zeros((batch_size, self.hidden_size), device=x.device)

        for t in range(sequence_len):
            x_t = x[:, t, :]

            i_t = self.sigmoid(self.W_ii(x_t) + self.W_hi(h))
            f_t = self.sigmoid(self.W_if(x_t) + self.W_hf(h))
            g_t = self.tanh(self.W_ig(x_t) + self.W_hg(h))
            o_t = self.sigmoid(self.W_io(x_t) + self.W_ho(h))
            c = f_t * c + i_t * g_t
            h = o_t * self.tanh(c)
            outputs.append(h)

        return torch.stack(outputs, dim=1)