import torch

#5 Функция
def grad_example5(x):
    x = x.clone()
    x.requires_grad_(True)

    f = torch.max(torch.abs(x))
    f.backward()

    return x.grad

#10 Функция
def pi(u):
    return torch.exp(u)/(torch.sum(torch.exp(u)))

def grad_example10(u):
    u = u.clone()
    u.requires_grad_(True)

    f = -torch.sum(pi(u) * torch.log(pi(u)))
    f.backward()

    return u.grad

#print(grad_example5(torch.tensor([-4.,3.,2.])))
print(grad_example10(torch.tensor([-4.,3.,2.])))