import logging
import torch.nn as nn
from torch.nn.modules.loss import PoissonNLLLoss
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None

    def set_learning_rate(self, learning_rate, optimizer_type:str="SGD"):
        logging.info('Current learning rate: %f,Optimizer Type %s', learning_rate, optimizer_type)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00)

        if optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00)
            pass
        elif optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-03)
            pass
        elif optimizer_type == "RMSprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate, alpha=0.9)
            pass
        else:
            print("Not support optimizer: ", optimizer_type)
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, drop_last=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):  #
            epoch_loss = 0
            for data in self.data_loader:
                inputs, values = data
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()  #Clears the gradients of all optimized torch.Tensors.
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, values)  #Compute MSELoss
                loss.backward()
                self.optimizer.step()  # updates the parameters after the  gradients are computed using e.g. backward().
                epoch_loss += loss.data.item(
                )  #Returns the value of this tensor as a standard Python number. This only works for tensors with one element. For other cases, see tolist().

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            inputs, values = next(iter(self.data_loader))
            inputs = Variable(inputs)
            values = Variable(values)

            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss
