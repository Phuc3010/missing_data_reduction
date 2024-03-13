from torch import nn, optim
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from torchvision.models import resnet50, resnet18
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}

class NeuralNet(nn.Module):
    def __init__(
        self,
        n_unit_in: int,
        categories_cnt: int,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        n_iter: int = 20,
        batch_size: int = 64,
        n_iter_print: int = 10,
        patience: int = 10,
        n_iter_min: int = 5,
        dropout: float = 0.1,
        clipping_value: int = 1,
        batch_norm: bool = False,
        early_stopping: bool = True,
    ) -> None:
        super(NeuralNet, self).__init__()

        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]

        if n_layers_hidden > 0:
            layers = [nn.Linear(n_unit_in, n_units_hidden), nn.LayerNorm(n_units_hidden), NL()]
            # add required number of layers
            for i in range(n_layers_hidden - 1):
                layers.extend(
                        [
                            nn.Dropout(dropout),
                            nn.Linear(n_units_hidden, n_units_hidden),
                            nn.LayerNorm(n_units_hidden),
                            NL(),
                        ]
                    )
                
            layers.append(nn.Linear(n_units_hidden, categories_cnt))
        else:
            layers = [nn.Linear(n_unit_in, categories_cnt)]

        # return final architecture
        self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(*layers).to(DEVICE)
        self.categories_cnt = categories_cnt

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        self.model.train()
        X = self._check_tensor(X).float()
        y = self._check_tensor(y).squeeze().long()

        dataset = TensorDataset(X, y)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=False, shuffle=True)

        # do training
        val_loss_best = 999999
        patience = 0

        loss = nn.CrossEntropyLoss()

        for i in range(self.n_iter):
            train_loss = []

            for batch_ndx, sample in enumerate(loader):
                self.optimizer.zero_grad()

                X_next, y_next = sample

                preds = self.forward(X_next).squeeze()

                batch_loss = loss(preds, y_next)

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    X_val, y_val = test_dataset.dataset.tensors

                    preds = self.forward(X_val).squeeze()
                    val_loss = loss(preds, y_val)

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience > self.patience and i > self.n_iter_min:
                            break

                    if i % self.n_iter_print == 0:
                        print(
                            f"Epoch: {i}, Validation loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
                        )
        return self
    
    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        return self.softmax(self.model(X)).detach().clone().numpy()

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)
    
    def score(self, X, y):
        X = self._check_tensor(X).float()
        predict = self.predict(X)
        predicted_classes = np.argmax(predict, axis=1)
        accuracy = np.mean(predicted_classes == y)
        return accuracy
    
class Resnet18Wrapper(object):
    def __init__(self, batch_size, epochs):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )
        
        self.model = resnet18(num_classes=100)
        self.model.float()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), momentum=0.9, lr=0.1, nesterov=True,\
                                   weight_decay=.0005)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3,
                                                              threshold=0.001, mode='max')
    
    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)
    
    def transforms_dataset(self, dataset):
        preprocessed_dataset = []
        for i in range(len(dataset)):
            image = self.transforms(dataset[i])
            preprocessed_dataset.append(image)
        preprocessed_dataset = torch.stack(preprocessed_dataset, dim=0)
        return preprocessed_dataset
    
    def fit(self, X, y):
        X = self.transforms_dataset(X)
        y= torch.tensor(y)
        y = y.long()
        dataset = TensorDataset(X, y)
        train_size = 45_000
        valid_size = 5000
        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset, [train_size, valid_size]
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,\
                                            pin_memory=True, num_workers=4, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size,\
                                                  pin_memory=True, num_workers=4, shuffle=False)
        self.model.train()
        best_loss = float('inf')
        for epoch in tqdm(range(self.epochs)):
            for i, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                valid_loss = 0.0
                for i, batch in enumerate(valid_loader):
                    X, y = batch
                    X, y = X.to(self.device), y.to(self.device)
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                    valid_loss += loss.item()
                valid_loss /= len(valid_loader)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(
                        self.model.state_dict(), 'resnet18_state_dict.pt'
                    )
                self.scheduler.step(valid_loss)
                
        return self
    
    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        pred = torch.tensor([])
        X = self.transforms_dataset(X)
        dataset = TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, pin_memory=True,
                                            num_workers=4, shuffle=False)
        loader_tqdm = tqdm(loader)
        for i, batch in enumerate(loader_tqdm):
            X = batch[0]
            X = X.to(self.device)
            predicts = F.softmax(self.model(X), dim=-1)
            pred = torch.concat((pred, predicts), dim=0)
        pred = pred.cpu().numpy()
        return pred
    
    @torch.no_grad()
    def score(self, X, y):
        predict = self.predict(X)
        predicted_classes = np.argmax(predict, axis=1)
        accuracy = np.mean(predicted_classes == y)
        return accuracy
    
class Classifier():
    def __init__(self):
        pass
    
    def get(self, name, *args, **kwargs):
        assert name in ['logistic', 'random_forest', 'neural_nets', 'resnet18']
        if name == 'logistic':
            return LogisticRegression(*args, **kwargs)
        elif name == 'random_forest':
            return RandomForestClassifier(*args, **kwargs)
        elif name == 'neural_nets':
            return NeuralNet(*args, **kwargs)
        elif name == 'resnet18':
            return Resnet18Wrapper(*args, **kwargs)
        else:
            return NotImplemented
    
    