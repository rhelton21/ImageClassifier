import argparse
from torchvision import models, datasets, transforms
from torch import nn, optim
import torch
from collections import OrderedDict

# Sample Run Command:
# python train.py --data_dir /path/to/data --arch densenet121 --learning_rate 0.001 --hidden_units 500 --save_dir /path/to/checkpoint --gpu --epochs 5
# python train.py --data_dir flowers --learning_rate 0.01 --arch "vgg13" --epochs 3 --gpu
# 


MODEL_DICT = {
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'vgg13': models.vgg13,
    'densenet121': models.densenet121,
}


def create_model(arch, class_to_index, hidden_units, learning_rate=0.001):
    print(f"Executing create_model with architecture: {arch}")

    try:
        model = MODEL_DICT[arch](pretrained=True)
        print(f"Successfully loaded pretrained model {arch}")
    except KeyError:
        print(f"Invalid architecture {arch}, defaulting to densenet121")
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    input_size = model.classifier.in_features if arch.startswith('densenet') else model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    model.class_to_index = class_to_index
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print("Finished executing create_model")
    return model, optimizer, criterion


def validation(model, dataloaders, criterion):
    print("Starting validation")
    test_loss = 0
    accuracy = 0
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    for images, labels in dataloaders['validloader']:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    print("Validation complete")
    return test_loss, accuracy


def save_checkpoint(arch, state_dict, class_to_idx, hidden_units):
    print("Saving checkpoint")
    checkpoint = {'arch': arch, 'class_to_idx': class_to_idx, 'state_dict': state_dict, 'hidden_units': hidden_units}
    print("Checkpoint saved")
    return checkpoint


def train_model(class_to_idx, dataloaders, arch, learning_rate, hidden_units, checkpoint):
    print("Starting model training")
    model, optimizer, criterion = create_model(arch, class_to_idx, hidden_units, learning_rate)
    epochs = args.epochs
    print_every = 40
    steps = 0
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders['trainloader']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloaders, criterion)

                print(f"Epoch: {e + 1}/{epochs}, Training Loss: {running_loss / print_every:.3f}, "
                      f"Validation Loss: {test_loss / len(dataloaders['validloader']):.3f}, "
                      f"Validation Accuracy: {accuracy / len(dataloaders['validloader']):.3f}")

                running_loss = 0
                model.train()

    if checkpoint:
        checkpoint_saved = save_checkpoint(arch, model.state_dict(), model.class_to_index, hidden_units)
        torch.save(checkpoint_saved, checkpoint)
        print(f'Checkpoint saved at {checkpoint}')

    print("Model training complete")
    return model


def main():
    print(f"Starting the program with the following parameters:")
    print(f"Data directory: {args.data_dir}")
    print(f"Architecture: {args.arch}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hidden units: {args.hidden_units}")
    print(f"Save directory: {args.save_dir}")
    print(f"GPU enabled: {args.gpu}")
    print(f"Number of epochs: {args.epochs}")
 
    print("Executing main")
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets = {
        "train_data": datasets.ImageFolder(train_dir, transform=train_transforms),
        "valid_data": datasets.ImageFolder(valid_dir, transform=validation_transforms),
        "test_data": datasets.ImageFolder(test_dir, transform=test_transforms)
    }

    dataloaders = {
        "trainloader": torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=32, shuffle=True),
        "validloader": torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32),
        "testloader": torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=32)
    }

    class_to_idx = image_datasets['train_data'].class_to_idx
    train_model(class_to_idx, dataloaders, arch=args.arch, learning_rate=args.learning_rate, hidden_units=args.hidden_units, checkpoint=args.save_dir)

    print("Finished executing main")


if __name__ == '__main__':
    print("Sample --> python train.py --data_dir flowers --learning_rate 0.01 --arch \"vgg13\" --epochs 3 --gpu")
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument("--data_dir", type=str, required=True, help='Path to folder of images')
    parser.add_argument('--arch', type=str, default='densenet121', help='Chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoint')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')

    args = parser.parse_args()
    main()
