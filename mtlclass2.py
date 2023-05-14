import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
from torchvision import transforms
import numpy as np

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
BASE_PATH = 'MAFood121/'

#Classes of dishes
f = open(BASE_PATH + '/annotations/dishes.txt', "r")
classes = f.read().strip().split('\n')
f.close()

#Base Ingredients
f = open(BASE_PATH + '/annotations/baseIngredients.txt', "r")
base_ing = f.read().strip().split(', ')
f.close()

# Load data from .h5 files
train_df = pd.read_hdf('train_df.h5', 'df')
val_df = pd.read_hdf('val_df.h5', 'df')
test_df = pd.read_hdf('test_df.h5', 'df')

epochs = 8
batch_size = 16
SMALL_DATA = False
IMG_SIZE = (224, 224)

if SMALL_DATA:
    train_df = train_df[:128]
    val_df = test_df[:128]
    test_df = test_df[:128]

col_names = list(train_df.columns.values)

ing_names = col_names[:-3]
targets = ing_names

class CustomDataset(Dataset):
    def __init__(target, df):
        target.df = df

    def __len__(target):
        return len(target.df)

    def __getitem__(target, idx):
        #print(target.df.iloc[idx])
        image_path = target.df.iloc[idx]['path']
        image = cv2.imread(image_path, 1)
        x = cv2.resize(image, IMG_SIZE)
        x = (torch.from_numpy(x.transpose(2,0,1))).float()
        sl_class_id = int(target.df.iloc[idx]['sl_class_id'])
        sl_onehot = np.array(sl_class_id)
        sl_onehot = (np.arange(len(classes)) == sl_onehot).astype(np.float32)
        sl_y = torch.from_numpy(sl_onehot)
        ml_y = []
        for i in range(len(base_ing)): # total food family
            ml_y.append(target.df.iloc[idx][str(i)])
        ml_y = np.array(ml_y, dtype=np.float32)
        #print(image_path, sl_y, ml_y)
        return (x, sl_y, ml_y)

# Define batch size
batch_size = 32

# Create DataLoader objects for training, validation, and testing sets
train_dataset = CustomDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(val_df)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = CustomDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ResNet50 Model
from torchvision import models
from torch import nn
from torchsummary import summary

resnet = models.resnet50(pretrained=True)
# Disable grad for all conv layers
for param in resnet.parameters():
    param.requires_grad = False

# Add two heads
resnet.last_linear = resnet.fc
n_features = resnet.fc.out_features
head_sl = nn.Sequential(
    nn.Linear(n_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(512, len(classes))
)
head_ml = nn.Sequential(
    nn.Linear(n_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(512, len(base_ing)),
    nn.Sigmoid()
)  
        
# Connect two heads
class FoodModel(nn.Module):
    def __init__(target, base_model, head_sl, head_ml):
        super().__init__()
        target.base_model = base_model
        target.head_sl = head_sl
        target.head_ml = head_ml

    def forward(target, x):
        x = target.base_model(x)
        sl = target.head_sl(x)
        ml = target.head_ml(x)
        return sl, ml

model = FoodModel(resnet, head_sl, head_ml)
#summary(model, (3, 224, 224))
model.to(device)

# Define Loss
sl_loss_fn = nn.CrossEntropyLoss()
ml_loss_fn = nn.BCELoss()

# Define Optimizer
optimizer = torch.optim.Adam(model.parameters())

# Define function to calculate loss and train model
def train_step(model, optimizer, sl_loss_fn, ml_loss_fn, data, device):
    # Retrieve data
    x, sl_y, ml_y = data
    # Convert to device
    x = x.to(device)
    sl_y = sl_y.to(device)
    ml_y = ml_y.to(device)
    # Zero out gradients
    optimizer.zero_grad()
    # Forward pass
    sl_preds, ml_preds = model(x)
    # Calculate losses
    sl_loss = sl_loss_fn(sl_preds, sl_y)
    ml_loss = ml_loss_fn(ml_preds, ml_y)
    loss = sl_loss + ml_loss
    # Backward pass
    loss.backward()
    # Step optimizer
    optimizer.step()
    # Return losses
    return sl_loss.item(), ml_loss.item()

epochs= 2
for i in tqdm(range(epochs), desc='Epochs'):
    print("Epoch ",i)
    with tqdm(train_loader, desc='Training', total=len(train_loader), miniters=1) as pbar:
        print(pbar)
        for data in pbar:
            print(train_step(model, optimizer, sl_loss_fn, ml_loss_fn, data, device))
            break
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load a test image
img_path = '82439.jpg'
img = Image.open(img_path).convert('RGB')
plt.imshow(img)

# Resize image and convert to tensor
transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
img = transform(img)
img = img.unsqueeze(0)

# Get model predictions
model.eval()
with torch.no_grad():
    sl_preds, ml_preds = model(img.to(device))

sl_preds = torch.nn.functional.softmax(sl_preds)
sl_preds = sl_preds.cpu().numpy()
ml_preds = ml_preds.cpu().numpy()

# Plot prediction results
sl_preds = sl_preds.squeeze()
plt.figure(figsize=(10, 5))
plt.bar(classes, sl_preds)
plt.title('Softmax Prediction')
plt.xlabel('Food Category')
plt.ylabel('Probability')
#plt.show()
plt.savefig("sl_result.jpg")

ml_preds = ml_preds.squeeze()
plt.figure(figsize=(10, 5))
plt.bar(base_ing, ml_preds)
plt.title('Sigmoid Prediction')
plt.xlabel('Ingredient')
plt.ylabel('Probability')
#plt.show()
plt.savefig("ml_result.jpg")