import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
nd= torch.distributions.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
print(nd.sample())
