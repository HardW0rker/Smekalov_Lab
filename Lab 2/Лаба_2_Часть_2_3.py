import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

batch_size = 128
learning_rate = 0.001
epochs = 3 #Количество эпох
mnt=0.5

input_size = 28*28
num_classes = 10

dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True) #Скачиваем дата сет

train_ds, val_ds = random_split(dataset, [50000, 10000])
test_ds = MNIST(root='data/', train=False, transform=transforms.ToTensor())

#Разбиваем на три выборки
train_loader = DataLoader(train_ds, batch_size, shuffle=True) 
val_loader = DataLoader(val_ds, batch_size*2)
test_loader = DataLoader(test_ds, batch_size*2)

image, label = train_ds[0]

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
       
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
   
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Получение предсказания
        loss = F.cross_entropy(out, labels) # Вычисление потерь
        return loss
   
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Получение предсказания
        loss = F.cross_entropy(out, labels)   # Вычисление потерь
        acc = accuracy(out, labels)           # Вычесление точности
        return {'Значение потери': loss.detach(), 'Точность': acc.detach()}
       
    def validation_epoch_end(self, outputs):
        batch_losses = [x['Значение потери'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Объединяем потери
        batch_accs = [x['Точность'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Объединяем точность
        return {'Значение потери': epoch_loss.item(), 'Точность': epoch_acc.item()}
   
    def epoch_end(self, epoch, result):
        print("Эпоха [{}], Значение потери: {:.4f}, Точность: {:.4f}".format(epoch, result['Значение потери'], result['Точность']))
   
model = MnistModel()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr, momentum=mnt)
    for epoch in range(epochs):
        # Этап обучения
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Этап валидации
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

evaluate(model, val_loader)
history = fit(epochs, 0.01, model, train_loader, val_loader)

#Выводим график количества потерь на разных эпохах
accuracies = [r['Значение потери'] for r in history]
plt.plot(accuracies)
plt.xlabel('Эпоха')
plt.ylabel('Значение потери')
plt.title('Количество потерь на разных эпохах')
plt.show()

result = evaluate(model, test_loader)

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

#Проверка предсказания
img, label = test_ds[100]
plt.imshow(img[0], cmap='gray')
print('Проверяем пресказание цифры')
print('Цифра:', label, ', Предсказание:', predict_image(img, model))
plt.show()

n = 10
m = 10

fail = 0
quantity_fail = 2 #Сколько нужно вывести неправльно распознах цифр
cls_err = [[0] * m for i in range(n)]
cls_data = [[],[]]

#Выводим несколько неправильно распознанных цифр
print('Выводим ',quantity_fail,' неправильно распознанных цифр')
for i in range(len(test_ds)):
    img, label = test_ds[i]
    prediction = predict_image(img, model)
    cls_data[0].append(label)
    cls_data[1].append(prediction)
    if label != prediction:
        cls_err[label][prediction]+=1
        if fail<quantity_fail:
            fail+=1
            plt.imshow(img[0], cmap='gray')
            print('Цифра:', label, ', Предсказание:', predict_image(img, model))
            plt.show()

#Выводим клацификацию ошибок
print('Выводим клацификацию ошибок')
print(classification_report(cls_data[0], cls_data[1]))
cls_err_np = np.array(cls_err)

def Creating_pairs(a):
    max_elem = a[0][0]
    for i in range(len(a)):
        for j in range(len(a[i] )):
             if a[i][j] > max_elem:
                   max_elem =  a[i][j]

    list_index_max =[ (i,j) for i in range(len(a))  for j in range(len(a[i])) if a[i][j]  == max_elem]
    line, column = list_index_max[0]
    return [line,column,max_elem]

def confusing_numbers(a):
    a = a + np.transpose(a)
    a = np.triu(a)
    conf_numbers = 3
    print('Выведем ',conf_numbers, ' цифр(ы), которые чаще всего путают')
    for i in range(conf_numbers):
        elem = Creating_pairs(a)
        print(f'Цифры {elem[0]} и {elem[1]} {elem[2]} количество раз')
        a[elem[0]][elem[1]]=0

confusing_numbers(cls_err_np)





