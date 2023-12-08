## References
## https://colab.research.google.com/drive/1CBIdPxHn_W2ARx4VozRLIptBrXk7ZBoM?usp=sharing#scrollTo=-Xon2PHWno7t
## https://colab.research.google.com/drive/1IUe9lfoIiQsL49atSOgxnCmMR_zJazKI#scrollTo=aZbW6Pj0og7K

import numpy as np
import torch

from torch import nn

device = torch.device('cuda')

class Three_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lcl_model = LSTM(input_size, hidden_size, num_layers, 1)
        self.lk_model = LSTM(input_size, hidden_size, num_layers, 1)
        self.lcr_model = LSTM(input_size, hidden_size, num_layers, 1)

    def forward(self, x):
        batch_size = x.size(0)
        pred = torch.zeros(batch_size, self.output_size).to(device)

        pred[:, 0] = torch.squeeze(self.lcl_model(x.detach().clone().to(device)), dim=1)
        pred[:, 1] = torch.squeeze(self.lk_model(x.detach().clone().to(device)), dim=1)
        pred[:, 2] = torch.squeeze(self.lcr_model(x.detach().clone().to(device)), dim=1)

        return pred
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        x, _ = self.lstm(x, (h0, c0))
        x = self.linear(x[:, -1, :])
        return x
    
class Three_HMM(nn.Module):
    def __init__(self, input_size, hidden_states, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_states = hidden_states
        self.output_size = output_size

        self.lcl_model = NHMM(input_size, hidden_states, 1)
        self.lk_model = NHMM(input_size, hidden_states, 1)
        self.lcr_model = NHMM(input_size, hidden_states, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        pred = torch.zeros(batch_size, self.output_size).to(device)

        pred[:, 0] = self.lcl_model(x.detach().clone().to(device))
        pred[:, 1] = self.lk_model(x.detach().clone().to(device))
        pred[:, 2] = self.lcr_model(x.detach().clone().to(device))

        return pred

class Three_LSTMHMM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_states, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_states = hidden_states
        self.output_size = output_size

        self.lcl_model = LSTMHMM(input_size, hidden_size, hidden_states, 1)
        self.lk_model = LSTMHMM(input_size, hidden_size, hidden_states, 1)
        self.lcr_model = LSTMHMM(input_size, hidden_size, hidden_states, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        pred = torch.zeros(batch_size, self.output_size).to(device)

        pred[:, 0] = self.lcl_model(x.detach().clone().to(device))
        pred[:, 1] = self.lk_model(x.detach().clone().to(device))
        pred[:, 2] = self.lcr_model(x.detach().clone().to(device))

        return pred

class NHMM(nn.Module):
    def __init__(self, input_size, hidden_states, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_states = hidden_states
        self.output_size = output_size

        self.transition_model = TransitionModel(hidden_states)
        self.emission_model = NeuralEmissionModel(input_size, hidden_states)
        self.unnormalized_state_priors = nn.Parameter(torch.randn(hidden_states)).to(device)
    
    def forward(self, x):
        batch_size = x.size(0)
        timesteps = x.size(1)

        log_state_priors = nn.functional.log_softmax(self.unnormalized_state_priors, dim=0).to(device)

        log_alpha = torch.zeros(batch_size, self.hidden_states).to(device)
        
        log_alpha = self.emission_model(x[:, 0, :]) + log_state_priors
        for t in range(1, timesteps):
            log_alpha = self.emission_model(x[:, t, :]) + self.transition_model(log_alpha)

        log_probs = log_alpha.logsumexp(dim=1)
        return log_probs
    
class LSTMHMM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_states, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_states = hidden_states
        self.output_size = output_size

        self.transition_model = TransitionModel(hidden_states)
        self.emission_model = LSTMEmissionModel(input_size, hidden_size, hidden_states)
        self.unnormalized_state_priors = nn.Parameter(torch.randn(hidden_states)).to(device)

    def forward(self, x):
        batch_size = x.size(0)
        timesteps = x.size(1)

        log_state_priors = nn.functional.log_softmax(self.unnormalized_state_priors, dim=0).to(device)

        log_alpha = torch.zeros(batch_size, self.hidden_states).to(device)

        emission = self.emission_model(x)
        
        log_alpha = emission[:, 0, :] + log_state_priors
        for t in range(1, timesteps):
            log_alpha = emission[:, t, :] + self.transition_model(log_alpha)

        log_probs = log_alpha.logsumexp(dim=1)
        return log_probs

class TransitionModel(nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states
        self.unnormalized_transition_matrix = nn.Parameter(torch.randn(hidden_states, hidden_states)).to(device)

    def forward(self, log_alpha):
        log_transition_matrix = nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)
        out = log_domain_matmul(log_transition_matrix, log_alpha.transpose(0, 1)).transpose(0, 1)

        return out

class NeuralEmissionModel(nn.Module):
    def __init__(self, input_size, hidden_states):
        super().__init__()
        self.neural_stack = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, hidden_states),
        )

    def forward(self, x):
        x = self.neural_stack(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
    
class GaussianMixModel(nn.Module):
    def __init__(self, n_features, n_components):
        super().__init__()
        self.init_scale = np.sqrt(6 / n_features)
        self.n_features = n_features
        self.n_components = n_components

        weights = torch.ones(n_components)
        means = torch.randn(n_components, n_features) * self.init_scale
        stdevs = torch.rand(n_components, n_features) * self.init_scale
        
        self.blend_weight = nn.Parameter(weights)
        self.means = nn.Parameter(means)
        self.stdevs = nn.Parameter(stdevs)
    
    def forward(self, x):
        blend_weight = torch.distributions.Categorical(nn.functional.relu(self.blend_weight))
        comp = torch.distributions.Independent(torch.distributions.Normal(self.means, torch.abs(self.stdevs)), 1)
        gmm = torch.distributions.MixtureSameFamily(blend_weight, comp)
        return -gmm.log_prob(x)

class LSTMEmissionModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_states):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_states = hidden_states
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_states)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        x, _ = self.lstm(x, (h0, c0))
        x = self.linear(x)
        x = nn.functional.log_softmax(x, dim=2)
        return x

def log_domain_matmul(log_A, log_B):
    m = log_A.size(0)
    n = log_A.size(1)
    p = log_B.size(1)

    log_A_expanded = torch.reshape(log_A, (m, n, 1))
    log_B_expanded = torch.reshape(log_B, (1, n, p))

    elementwise_sum = log_A_expanded + log_B_expanded
    out = torch.logsumexp(elementwise_sum, dim=1)

    return out
