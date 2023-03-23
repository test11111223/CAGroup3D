import torch


a = torch.randn(85, 3)

b = a.new_zeros(len(a))

indices1 = torch.nonzero(torch.randn(155) * 155).squeeze(-1)
indices = [ac.long() for ac in indices1 if ac < len(a)]   

b[indices] = 1

print(b)

c = torch.zeros_like(a[:, :3])

print(c.size())

c[indices, :] = torch.randn(3)

#print(c)

d =  a.new_zeros(len(a))

print(d.size())

d[indices] 

print(max(indices))