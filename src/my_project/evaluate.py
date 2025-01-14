import torch
from data import corrupt_mnist
from model import MyAwesomeModel
import typer

def evaluate(model_checkpoint: str) -> None: 
    """Loads a saved model and evaluates it"""

    # Loads the model and the weights
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    # Loads the test set
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    # calculates the accuracy of the test set
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():

        for img, target in test_dataloader:
            # predict the class
            pred = model(img)
            
            correct += (pred.argmax(dim=1) == target).float().sum().item() # check if correct
            total += target.size(0)

    print(f'Test accuracy: {100*correct / total}')


if __name__ == '__main__':
    typer.run(evaluate)