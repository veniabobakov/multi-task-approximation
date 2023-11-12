import torch
import matplotlib.pyplot as plt


class GenData:
    """
    Класс для генерации обучающей и валидационной выборки для аппроксимации элементарных функций
    n - количество точек в тестовом наборе
    func - список функций вида torch.nameOfFunc
    На выходе получаем обучающие и валлидационные выборки созначениями х и у изучаемых функций
    """

    def __init__(self, n: int, last: bool, *func) -> None:
        # Data for train
        self.numFunc = len(func)
        x_train = torch.rand(n)
        x_train = x_train * 40.0 - 20.0
        x_train, indices = torch.sort(x_train)
        self.x_train = x_train[(x_train <= -4) | (x_train >= 4)]
        arr = []
        noise = torch.zeros(len(self.x_train))
        for i, f in enumerate(func):
            add = f(self.x_train)
            m = torch.max(add)
            r = torch.rand(len(self.x_train)) * (m / 2) - (m / 4)
            if i == len(func) - 1 and last:
                arr.append(add + noise)
            else:
                arr.append(add + r)
            if last:
                noise += r
        self.y_train = torch.column_stack(tuple(arr))

        # Validate data
        self.x_val = torch.linspace(-60, 60, 600)
        arr = []
        for f in func:
            arr.append(f(self.x_val))
        self.y_val = torch.column_stack(tuple(arr))

    def get_train_unsqueeze(self):
        return torch.unsqueeze(self.x_train, 1), torch.unsqueeze(self.y_train, 1)

    def get_val_unsqueeze(self):
        return torch.unsqueeze(self.x_val, 1), torch.unsqueeze(self.y_val, 1)

    def plot_train_signals(self):
        fig, axs = plt.subplots(self.numFunc, figsize=(12, 5 * self.numFunc))
        for i in range(self.numFunc):
            axs[i].scatter(self.x_train.numpy(), self.y_train[:, i].numpy())

    def plot_val_signals(self):
        fig, axs = plt.subplots(self.numFunc, figsize=(20, 5 * self.numFunc))
        for i in range(self.numFunc):
            axs[i].plot(self.x_val.numpy(), self.y_val[:, i].numpy())


if __name__ == "__main__":
    pass
