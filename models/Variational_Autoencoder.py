import torch
import torch.nn as nn


class Variational_LSTM_Audoencoder(nn.Module):
    def __init__(self):
        super(Variational_LSTM_Audoencoder, self).__init__()
        