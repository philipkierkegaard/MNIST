from model import MyAwesomeModel
from data import corrupt_mnist
import typer

import torch
from torch import optim
import matplotlib.pyplot as plt

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size)

    stats = {'train_loss': [], 'train_accuracy': []}

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        for i, (images, targets) in enumerate(train_dataloader):

            optimizer.zero_grad() # set gradient to zero because it accumulate
            preds = model(images) # use the model to predict the label
            loss = criterion(preds, targets) # calculate the loss based on the criterion
            loss.backward() # use the loss to calculate the gradient
            optimizer.step() # update the weights based on the gradient

            stats['train_loss'].append(loss.item()) # append the loss

            # calculate the accuracy and append it
            accuracy = (preds.argmax(dim=1)==targets).float().mean().item() 
            stats['train_accuracy'].append(accuracy)

            # every 100 batch, print stats
            if i % 400 == 0:
                print(f'Epoch: {epoch}, accuracy: {accuracy}, loss: {loss.item()}')

    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(stats["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(stats["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == '__main__':
    typer.run(train)