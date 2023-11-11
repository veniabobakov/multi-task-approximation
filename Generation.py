import torch


class GenData:
    """
    Класс для генерации обучающей и валидационной выборки для аппроксимации элементарных функций
    func - список функций вида torch.nameOfFunc
    На выходе получаем обучающие и валлидационные выборки созначениями х и у изучаемых функций
    """

    def __int__(self, n, *func):
        # Data for train
        x_train = torch.rand(n)
        x_train = x_train * 40.0 - 20.0

        self.x_train = x_train[(x_train <= -4) | (x_train >= 4)]
        arr = []
        for f in func:
            arr.append(f(self.x_train))
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
