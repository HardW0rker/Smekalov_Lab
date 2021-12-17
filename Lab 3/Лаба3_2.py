import torch

def zi(x,w,b):
    return x*w+b 

def sigma(x,w,b):
    return 1/(1 + torch.exp(-zi(x,w,b)))

def foo():
    X = torch.tensor([[ 0.4700,  5.5100],
        [-1.4300, -0.1300],
        [ 1.4000, -5.6100],
        [ 0.8600,  0.8200],
        [ 3.7600, -0.4100],
        [ 1.3800,  1.1900],
        [-1.5500, -0.0500],
        [-3.0400, -3.3000]])
    w = torch.randn(2, requires_grad=True)
    b = torch.tensor(0., requires_grad=True)
    y = torch.torch.tensor([[0],[0], [1], [1], [1], [1], [0], [1]])
    lr = 0.05 
    for iteration in range(100):
        with torch.no_grad():
            if w.grad is not None: 
                w.grad.zero_() 
            if b.grad is not None: 
                b.grad.zero_() 


        f = - torch.sum(torch.log(sigma(X,w,b)) * (y) + (torch.log((1 - sigma(X,w,b))) * (1-y)))
        if iteration % 10 == 0:
            print(w.data.tolist(), f.item())
            print(b.data.tolist(), f.item()) 
        f.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
    print(w.data,b.data,f.item())

foo()