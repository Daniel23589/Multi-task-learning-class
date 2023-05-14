import os
import torch
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra

print(os.listdir("MAFood121"))

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42) # try and make the results more reproducible
BASE_PATH = 'MAFood121/'

epochs = 35
batch_size = 64
MICRO_DATA = True # very small subset (just 3 groups)
SAMPLE_TRAINING = False # make train set smaller for faster iteration
IMG_SIZE = (384, 384) # Try to change the model to U-net to avoid the resizing

#Classes of dishes
f = open(BASE_PATH + '/annotations/dishes.txt', "r")
classes = f.read().strip().split('\n')
f.close()
print("***** classes = dishes.txt: ***** " + str(classes))
print("#######################################################################################")

#Ingredients for each class
f = open(BASE_PATH + '/annotations/foodgroups.txt', "r")
ingredients = list(set(f.read().strip().split('\n')))
f.close()
print("***** ingredients = foodgroups.txt: ***** " + str(ingredients))
print("#######################################################################################")

#Base Ingredients
f = open(BASE_PATH + '/annotations/baseIngredients.txt', "r")
base_ing = f.read().strip().split(', ')
f.close()
print("***** base_ing = baseIngredients.txt: ***** " + str(base_ing))
print("#######################################################################################")

#Recovery of annotations ML
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

#train
f = open(BASE_PATH + '/annotations/train.txt', "r")
train_images = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/train_lbls_ff.txt', "r")
train_labels = f.read().split('\n')
f.close()

#val
f = open(BASE_PATH + '/annotations/val.txt', "r")
val_images = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/val_lbls_ff.txt', "r")
val_labels = f.read().split('\n')
f.close()

#test
f = open(BASE_PATH + '/annotations/test.txt', "r")
test_images = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/test_lbls_ff.txt', "r")
test_labels = f.read().split('\n')
f.close()

#Recovery of annotations SL
#train
f = open(BASE_PATH + '/annotations/train.txt', "r")
train_imagessl = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/train_lbls_d.txt', "r")
train_labelssl = f.read().split('\n')
f.close()

#val
f = open(BASE_PATH + '/annotations/val.txt', "r")
val_imagessl = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/val_lbls_d.txt', "r")
val_labelssl = f.read().split('\n')
f.close()

#test
f = open(BASE_PATH + '/annotations/test.txt', "r")
test_imagessl = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/test_lbls_d.txt', "r")
test_labelssl = f.read().split('\n')
f.close()

# Single-Label
train_images_sl = ["MAFood121/images/" + s for s in train_imagessl]
train_df_sl = pd.DataFrame({'path': train_images_sl, 'sl_class_id': train_labelssl})

val_images_sl = ["MAFood121/images/" + s for s in val_imagessl]
val_df_sl = pd.DataFrame({'path': val_images_sl, 'sl_class_id': val_labelssl})

test_images_sl = ["MAFood121/images/" + s for s in test_imagessl]
test_df_sl = pd.DataFrame({'path': test_images_sl, 'sl_class_id': test_labelssl})

# Multi-label
train_images_ml = ["MAFood121/images/" + s for s in train_images]
train_df_ml = pd.DataFrame({'path': train_images_ml, 'ml_class_id': train_labels})

val_images_ml = ["MAFood121/images/" + s for s in val_images]
val_df_ml = pd.DataFrame({'path': val_images_ml, 'ml_class_id': val_labels})

test_images_ml = ["MAFood121/images/" + s for s in test_images]
test_df_ml = pd.DataFrame({'path': test_images_ml, 'ml_class_id': test_labels})
"""
# Train images
train_img_df = pd.concat([train_df_sl, train_df_ml])
val_img_df = pd.concat([val_df_sl, val_df_ml])
test_img_df = pd.concat([test_df_sl, test_df_ml])

train_img_df = train_img_df.sample(frac=1).reset_index(drop=True)
val_img_df = val_img_df.sample(frac=1).reset_index(drop=True)
test_img_df = test_img_df.sample(frac=1).reset_index(drop=True)

train_img_df['class_name'] = train_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
print(train_img_df)
print("-------------------------------------------------------------------------------------------------")

val_img_df['class_name'] = val_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
print(val_img_df)
print("-------------------------------------------------------------------------------------------------")

test_img_df['class_name'] = test_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
print(test_img_df)
"""
#Dataframe for train images
import glob

train_ingredients = []
train_classid = []

# busca ambos archivos en el directorio de anotaciones
for file_path in glob.glob(BASE_PATH + '/annotations/train_lbls_*.txt'):
    with open(file_path) as f1:
        for line in f1:
            idx_ingredients = []
            classid = int(line)
            train_classid.append(classid)
            for ing in ingredients[classid].strip().split(","):
                idx_ingredients.append(str(base_ing.index(ing)))
            train_ingredients.append(idx_ingredients)

df_train = pd.DataFrame(mlb.fit_transform(train_ingredients), columns=mlb.classes_) #binary encode ingredients
df_train["path"] = train_df_ml['path'] #train_img_df['path']
df_train["ml_class_id"] = train_classid 
food_dict_train = df_train


new_data = []
for index, row in train_df_ml.iterrows():
    #food = row["class_name"]
    path = row["path"]
    class_id = row["ml_class_id"]
    
    binary_encod = food_dict_train.loc[food_dict_train["path"] == path]
    new_data.append(np.array(binary_encod)[0])

col_names = list(binary_encod.columns.values)
train_df = pd.DataFrame(new_data, columns = col_names)

#Dataframe for val images

val_ingredients = []
val_classid = []

# busca ambos archivos en el directorio de anotaciones
for file_path in glob.glob(BASE_PATH + '/annotations/val_lbls_*.txt'):
    with open(file_path) as f1:
        for line in f1:
            idx_ingredients = []
            classid = int(line)
            val_classid.append(classid)
            for ing in ingredients[classid].strip().split(","):
                idx_ingredients.append(str(base_ing.index(ing)))
            val_ingredients.append(idx_ingredients)

df_val = pd.DataFrame(mlb.fit_transform(val_ingredients), columns=mlb.classes_) #binary encode ingredients
df_val["path"] = val_df_ml['path']
df_val["ml_class_id"] = val_classid 
food_dict_val = df_val


new_data = []
for index, row in val_df_ml.iterrows():
    #food = row["class_name"]
    path = row["path"]
    class_id = row["ml_class_id"]
    
    binary_encod = food_dict_val.loc[food_dict_val["path"] == path]
    new_data.append(np.array(binary_encod)[0])

col_names = list(binary_encod.columns.values)
val_df = pd.DataFrame(new_data, columns = col_names)

#Dataframe for test images

test_ingredients = []
test_classid = []

# busca ambos archivos en el directorio de anotaciones
for file_path in glob.glob(BASE_PATH + '/annotations/test_lbls_*.txt'):
    with open(file_path) as f1:
        for line in f1:
            idx_ingredients = []
            classid = int(line)
            test_classid.append(classid)
            for ing in ingredients[classid].strip().split(","):
                idx_ingredients.append(str(base_ing.index(ing)))
            test_ingredients.append(idx_ingredients)

df_test = pd.DataFrame(mlb.fit_transform(test_ingredients), columns=mlb.classes_) #binary encode ingredients
df_test["path"] = test_df_ml['path']
df_test["ml_class_id"] = test_classid 
food_dict_test = df_test


new_data = []
for index, row in test_df_ml.iterrows():
    #food = row["class_name"]
    path = row["path"]
    class_id = row["ml_class_id"]
    
    binary_encod = food_dict_test.loc[food_dict_test["path"] == path]
    new_data.append(np.array(binary_encod)[0])

col_names = list(binary_encod.columns.values)
test_df = pd.DataFrame(new_data, columns = col_names)

train_df = train_df.merge(train_df_sl, left_on='path', right_on='path')
val_df = train_df.merge(val_df_sl, left_on='path', right_on='path') 
test_df = train_df.merge(val_df_sl, left_on='path', right_on='path')

train_df.to_hdf('train_df.h5','df',mode='w',format='table',data_columns=True)
val_df.to_hdf('val_df.h5','df',mode='w',format='table',data_columns=True)
test_df.to_hdf('test_df.h5','df',mode='w',format='table',data_columns=True)