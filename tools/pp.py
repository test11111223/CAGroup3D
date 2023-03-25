import torch

a = torch.randn(97088,3)
b = a.new_zeros(len(a) + 1).to(a.device)   

print(a.size(), b.size())

c = [torch.tensor(72402, dtype=torch.long), torch.tensor(74158, dtype=torch.long)]

print(c)

# Error
#b[c] = 1

d = [i.item() for i in c]   

print(d)

b[d] = 1

print(1)