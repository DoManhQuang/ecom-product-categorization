import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# import torch.backends.cudnn as cudnn
# import numpy as np
# import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
# from PIL import Image
# from tempfile import TemporaryDirectory
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
import pandas as pd
import glob
from tqdm import tqdm


def save_training(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    length = len(glob.glob(save_dir + "/train*"))
    save_dir = os.path.join(save_dir, f'train{length+1}')
    os.makedirs(save_dir, exist_ok=True)
    print(">>>>>>>>>> ", save_dir)
    return save_dir


def train_model(model, criterion, optimizer, scheduler, tempdir, dataloaders, dataset_sizes, device='cpu', num_epochs=25):
    since = time.time()

    save_kfold_path = save_training(tempdir)

    # save training checkpoints
    best_model_params_path = os.path.join(save_kfold_path, 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print("Task : ", phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for _, (inputs, labels) in tqdm(enumerate(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


def load_dataset(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names


def train_fold(model_ft, data_dir, tempdir, epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device >>>> ", device)
    model_ft = model_ft.to(device)

    dataloaders, dataset_sizes, _ = load_dataset(data_dir)
    
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model=model_ft, criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, 
                tempdir=tempdir, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device, num_epochs=epochs)
    model.eval()

    
import shutil
def move_file(source_folder, destination_folder, file_name):
    # Check if the source file exists
    source_path = os.path.join(source_folder, file_name)
    if not os.path.exists(source_path):
        print(f"Source: {source_path}, file '{file_name}' not found in '{source_folder}'.")
        return
    
    # Check if the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # Create the destination folder if it doesn't exist
    
    # Construct the destination path
    destination_path = os.path.join(destination_folder, file_name)
    
    try:
        shutil.move(source_path, destination_path)
        print(f"File '{file_name}' moved from '{source_folder}' to '{destination_folder}'.")
    except Exception as e:
        print(f"Failed to move the file: {e}")


def move_file_kfold(dirs_source, dest_folder, data_path):
    for path, labels in data_path.values:
        move_file(source_folder=os.path.join(dirs_source, os.path.dirname(path)), 
                  destination_folder=os.path.join(dest_folder, os.path.dirname(path)),
                  file_name=os.path.basename(path))


def get_pretrained_model(model_name, weights, n_classes=1000):

    if model_name == 'vgg16':
        model = models.vgg16(weights=weights)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    return model


# data_dir = '/work/quang.domanh/ecom-product-categorization/Data_Folds/hova1000/KFold0'
epochs = 25

dir_image = "/work/quang.domanh/datasets/HOVA-1000/HOVA"
save_path = "/work/quang.domanh/ecom-product-categorization/Data_Folds/hova1000"
save_kfold = "/work/quang.domanh/ecom-product-categorization/logs/hova1000"
weights = VGG16_Weights.IMAGENET1K_V1

for i in range(5):
    if i != 0:
        continue
    # load data frame
    train_csv = pd.read_csv(os.path.join(save_path, f"KFold{i}", "train_path.csv"))
    test_csv = pd.read_csv(os.path.join(save_path, f"KFold{i}", "test_path.csv"))

    # move data
    data_dir = os.path.join(save_path, f"KFold{i}")
    os.makedirs(data_dir, exist_ok=True)
    # move_file_kfold(dirs_source=dir_image, dest_folder=data_dir, data_path=train_csv)
    # move_file_kfold(dirs_source=dir_image, dest_folder=data_dir, data_path=test_csv)

    
    # create save training
    tempdir = os.path.join(save_kfold, "vgg16", f"KFold{i}")
    os.makedirs(tempdir, exist_ok=True)
    
    # model = vgg16(weights=weights)

    # for param in model.parameters():
    #     param.requires_grad = False
    # model = get_pretrained_model(model_name='vgg16', weights=weights, n_classes=1000)

    # train_fold(data_dir=data_dir, model_ft=model, tempdir=tempdir, epochs=25)

    move_file_kfold(dirs_source=data_dir, dest_folder=dir_image, data_path=train_csv)
    move_file_kfold(dirs_source=data_dir, dest_folder=dir_image, data_path=test_csv)

