import hydra
from omegaconf import OmegaConf
import logging
import click
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Model
from data import CorruptMnist

logger = logging.getLogger(__name__)


# @click.group()
# def cli():
#     pass

# @click.command()
# @click.option("--lr", default=1e-5, help='learning rate')
@hydra.main(config_path="../config", config_name='config_default.yaml')
def main(config):
    # load config files
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hyparams_model = config.model
    hyparams_train = config.train
    hyparams_data = config.dataset

    model = Model(hyparams_model["x_dim"], hyparams_model["hidden_dim_1"], hyparams_model["hiden_dim_2"], hyparams_model["latent_dim"])
    train_set, test_set = CorruptMnist(train=True), CorruptMnist(train=False)
    trainloader = DataLoader(train_set, batch_size=hyparams_train["batch_size"])
    testloader = DataLoader(test_set, batch_size=len(test_set))

    epochs = 10
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), hyparams_train["lr"])
    
    train_losses = []
    test_losses = []

    print("=" * 50, "Training:", "=" * 50)
    for e in range(epochs):

        # training
        epoch_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad() # reset gradients
            output = model(images.float()) # run data through model
            loss = criterion(output, labels) # calculate loss
            loss.backward() # calculate gradient
            optimizer.step() # back probagate
            epoch_loss += loss.item() # calculate loss for each batch
        else:
            loss_epoch_mean = epoch_loss/len(trainloader)
            train_losses.append(loss_epoch_mean) # add mean loss for each epoch
            
        # validation
        with torch.no_grad():
            model.eval()
            for images, labels in testloader: # only run once
                images = images.view(images.shape[0], -1)
                output = model(images.float())
                loss = criterion(output, labels) # calculate loss
                test_loss = loss.item()/len(testloader) # calculate loss
                test_losses.append(test_loss)

    print("=" * 50, "Training  Done", "=" * 50)

    print("saving model")
    torch.save(model.state_dict(), 'model_checkpoint.pth')



# @click.command()
# @click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    model = Model()
    model.load_state_dict(torch.load(model_checkpoint))

    test_set = CorruptMnist(train=False)
    testloader = DataLoader(test_set, batch_size=len(test_set))

    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.shape[0], -1)
            output = model(images.float())
            _, top_class = output.topk(1, dim=1)
            top_class = torch.squeeze(top_class)
            equals = top_class == labels
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'Accuracy: {accuracy.item()*100}%')


# cli.add_command(train)
# cli.add_command(evaluate)
if __name__ == "__main__":
    main()