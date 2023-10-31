import numpy as np
import torch
from torchvision import models
from torch import nn
import argparse
from PIL import Image
import json
from collections import OrderedDict


# Sample Run Command:
# python predict.py --image flowers/test/4/image_05678.jpg --checkpoint checkpoint.pth --top_k 5 --labels cat_to_name.json --gpu

from collections import OrderedDict  # Don't forget to import OrderedDict

def load_checkpoint(filepath):
    """
    Load a pretrained model from a checkpoint file.
    """
    print("Executing load_checkpoint...")

    checkpoint = torch.load(filepath)

    arch = checkpoint['arch']
    print(f"Loading model with architecture: {arch}")

    model = models.__dict__[arch](pretrained=True)
    
    print("Freezing parameters so we don't backprop through them")
    for param in model.parameters():
        param.requires_grad = False
    
    # Find the first Linear layer in the Sequential object and get in_features from it.
    in_features = None
    if isinstance(model.classifier, nn.Sequential):
        for layer in model.classifier.children():
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                break
    elif isinstance(model.classifier, nn.Linear):  # if classifier is a single layer
        in_features = model.classifier.in_features
    else:
        raise ValueError("model.classifier does not contain any Linear layer")
    
    if in_features is None:
        raise ValueError("Could not find a Linear layer in the classifier")

    # Re-creating the classifier with the loaded checkpoint parameters
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, checkpoint['hidden_units'])),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    print("Finished executing load_checkpoint")
    return model

def process_image(image):
    """
    Process an image and convert it to a tensor.
    """
    print("Executing process_image...")
    
    # Assume image is a PIL.Image object
    # Convert the image to RGB (3 channels)
    image = image.convert('RGB')
    
    # Resize the image and keep the aspect ratio
    size = 256, 256
    image.thumbnail(size)
    
    # Crop the center of the image
    width, height = image.size   # Get dimensions
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    image = image.crop((left, top, right, bottom))
    
    # Convert image values to be between 0 and 1
    np_image = np.array(image)/255.0
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))
    
    print("Finished executing process_image")
    return torch.from_numpy(np_image).type(torch.FloatTensor)



def predict(image_path, model, topk, labels):
    """
    Predict the class of an image and print top probable classes.
    """
    print("Executing predict...")
    
    # Load label names (e.g. flower names)
    with open(labels, 'r') as f:
        cat_to_name = json.load(f)
    
    # Set model to evaluation mode
    model.eval()
    
    # Load and process the image
    img = Image.open(image_path)
    img = process_image(img)
    img.unsqueeze_(0)  # add batch dimension
    
    # Make a prediction
    with torch.no_grad():
        output = model.forward(img)
    
    # Convert log-probabilities to probabilities
    ps = torch.exp(output)
    prob, idxs = ps.topk(topk)  # get the top-k probabilities and indices
    
    # Map indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # Added Debugging print statements
    print(f'idx_to_class: {idx_to_class}')
    print(f'idxs[0]: {idxs[0]}')
    print(f'Elements in idxs[0]: {[idx.item() for idx in idxs[0]]}')
    
    try:
        top_classes = [idx_to_class[idx.item()] for idx in idxs[0]]  # Convert indices to class labels
        top_flowers = [cat_to_name[cls] for cls in top_classes]  # Convert class labels to flower names
    except KeyError as e:
        print(f"KeyError: {e}. The key does not exist in idx_to_class.")
        top_classes = []
        top_flowers = []
    
    print("Finished executing predict")
    return prob, top_classes, top_flowers


def main(args):
    """
    Main execution function.
    """
    print(f"Starting the program with the following parameters:")
    print(f"Image: {args.image}")
    print(f"Top K: {args.top_k}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Labels: {args.labels}")
    print(f"GPU enabled: {args.gpu}")

    print("Executing main...")
    model = load_checkpoint(args.checkpoint)
    prob, classes, flowers = predict(args.image, model, args.top_k, args.labels)
    
    print(f'Top {args.top_k} Classes and Probabilities: {list(zip(classes, prob[0]))}')
    print(f'Top {args.top_k} Flowers and Probabilities: {list(zip(flowers, prob[0]))}')
    
    print("Finished executing main")


if __name__ == '__main__':
    print("Script Started")
    print("Sample --> python predict.py --image flowers/test/4/image_05678.jpg --checkpoint checkpoint.pth --top_k 5 --labels cat_to_name.json --gpu")
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
    parser.add_argument('--image', type=str, required=True, help='Path to the test image.')
    parser.add_argument('--top_k', type=int, default=5, help='Top K most probable classes to return.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--labels', type=str, default='cat_to_name.json', help='Path to the file for label names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available.')
    
    args = parser.parse_args()
    
    main(args)
    
    print("Script Ended")
