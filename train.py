import os, cv2
import tensorflow as tf
# from model.rep_efficientnet_v2 import EfficientNetV2B3
import keras
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from model.losses import categorical_focal_crossentropy
# from keras.utils import image_dataset_from_directory
import tensorflow as tf
import glob
import keras
import numpy as np
# from keras import Sequential, Model
# from keras.layers import Input, Layer, AveragePooling2D, Conv2D, BatchNormalization, Dropout, Dense, Flatten, ReLU, ZeroPadding2D
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import shutil
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


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


def unfreeze_model(model, unfree=False, precent=10):
    length = len(model.layers)
    top_layer = int(length * (precent/100))
    print("length : ", length, "top_layer: ", top_layer, "int(precent/100): ", precent/100)
    if unfree:
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        top_layer *= -1
        for layer in model.layers[top_layer:]:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
                print("Layer >> ", layer)
    return model


def load_path_images(dir_images, read_imgs=False, gray=False, dsize=(224, 224)):
    tasks = ('train', 'val')
    data, labels, tmp_paths = [], [],[]
    for _, task in tqdm(enumerate(tasks)):
        cates = os.listdir(os.path.join(dir_images, task))
        for cate in cates:
            files = os.listdir(os.path.join(dir_images, task, cate))
            for img_file in files:
                labels.append(cate)
                tmp_path = os.path.join(dir_images, task, cate, img_file)
                tmp_paths.append(os.path.join(task,cate,img_file))
                
                if read_imgs:
                    image = cv2.imread(tmp_path)
                    image = cv2.resize(image, dsize)
                    if gray:
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        data.append(gray_image)
                    else:
                        data.append(image)
    return data, labels, tmp_paths


def load_images(dir_images, labels, gray=False, dsize=(224, 224)):
    print("LOAD IMAGE FROM DIR <<<< ", dir_images)
    data_imgs, data_labels = [], []
    for _, (dir_image, label) in tqdm(enumerate(zip(dir_images, labels))):
        if os.path.exists(dir_image):
            image = cv2.imread(dir_image)
            if image is not None:
                image = cv2.resize(image, dsize)
                if gray:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    data_imgs.append(gray_image)
                else:
                    data_imgs.append(image)
                data_labels.append(label)
    return np.array(data_imgs), np.array(data_labels)


def scale_up_lora(inputs, units, lora_rank, lora_num_layer, activation=None, name='lora'):
    lora_denses = []
    for i in range(lora_num_layer):
        lora_denses.append(keras.layers.Dense(units, activation=activation,
                                              lora_rank=lora_rank, name=f'{name}_dense_{i}')(inputs))
    lora_denses.append(inputs)
    x_nlora = keras.layers.Add(name=f'{name}_add')(lora_denses)
    # x_activation = keras.activations.relu(x_nlora)
    x_bn = keras.layers.BatchNormalization()(x_nlora)
    return x_bn


def build_model(base_model, size=(224, 224, 3), num_classs=1000, training=False):
    inputs = keras.Input(shape=size)
    x_base_model = base_model(inputs, training=training)
    x_flatten = keras.layers.GlobalAveragePooling2D()(x_base_model)
    x_dropout1 = keras.layers.Dropout(0.3)(x_flatten)
    dense = keras.layers.Dense(1024, activation='relu')(x_dropout1)
    outputs = keras.layers.Dense(num_classs, activation='softmax')(dense)
    model = keras.Model(inputs, outputs)
    return model


def save_training(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    length = len(glob.glob(save_dir + "/train*"))
    save_dir = os.path.join(save_dir, f'train{length+1}')
    os.makedirs(save_dir, exist_ok=True)
    print(">>>>>>>>>> ", save_dir)
    return save_dir


def get_cacll_back(save_dir):
    checkpoint_filepath = os.path.join(save_dir, 'best.weights.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True)

    model_stop = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                            verbose=1,
                                            patience=10,
                                            mode='max',
                                            restore_best_weights=False),
    return [
        model_stop,
        model_checkpoint_callback
    ]


def move_file_kfold(dirs_source, dest_folder, data_path):
    for path, labels in data_path.values:
        move_file(source_folder=os.path.join(dirs_source, os.path.dirname(path)), 
                  destination_folder=os.path.join(dest_folder, os.path.dirname(path)),
                  file_name=os.path.basename(path))


def train_kfold(model, dir_image, save_path, save_kfold, gray=False, size=(224, 224), batch_size=128):
    color_mode='rgb'
    if gray:
        color_mode='grayscale'

    # model = build_model(base_model=base_model,size=(224, 224))
    init_weights = model.get_weights()

    save_kfold_path = save_training(save_kfold)
    print(save_kfold_path)


    lb = preprocessing.LabelBinarizer()
    batch_size = 128
    epoch = 20

    for i in range(5):
        # if i != 0:
        #     continue
        train_csv = pd.read_csv(os.path.join(save_path, f"KFold{i}", "train_path.csv"))
        test_csv = pd.read_csv(os.path.join(save_path, f"KFold{i}", "test_path.csv"))

        move_file_kfold(dirs_source=dir_image, dest_folder=os.path.join(save_path, f"KFold{i}"), data_path=train_csv)
        move_file_kfold(dirs_source=dir_image, dest_folder=os.path.join(save_path, f"KFold{i}"), data_path=test_csv)

        # train_img_path = dir_image + "/" + train_csv['path'].astype(str).to_numpy()
        # train_labels_csv = train_csv['label'].to_numpy()
        # data_train, data_labels_train = load_images(dir_images=train_img_path, labels=train_labels_csv, gray=gray)
        # labels_one_hot_train = lb.fit_transform(data_labels_train)

        # test_img_path = dir_image + "/" + test_csv['path'].astype(str).to_numpy()
        # test_label_csv = test_csv['label'].to_numpy()
        # data_test, data_labels_test = load_images(dir_images=test_img_path, labels=test_label_csv, gray=gray)
        # labels_one_hot_test = lb.fit_transform(data_labels_test)



        # print(f"Fold {i}:")
        # print("train : ", data_train.shape, labels_one_hot_train.shape)
        # print("val : ", data_test.shape, labels_one_hot_test.shape)

        
        train = keras.utils.image_dataset_from_directory(
            directory=os.path.join(save_path, f"KFold{i}", "train"),
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=size,
            shuffle=True,
        )
        val = keras.utils.image_dataset_from_directory(
            directory=os.path.join(save_path, f"KFold{i}", "val"),
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=size,
            shuffle=True,
        )

        focal = keras.losses.categorical_focal_crossentropy
        METRICS = [
            tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='Top1'),
            # tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='Top5'),
        ]

        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005), loss=[focal], metrics = METRICS)
        my_callbacks = get_cacll_back(os.path.join(save_kfold_path, f"KFold{i}"))

        model.set_weights(init_weights)

        # # history = model.fit(data_train, labels_one_hot_train,
        # #         epochs = epoch,
        # #         validation_data = (data_test, labels_one_hot_test),
        # #         batch_size=batch_size,
        # #         shuffle=True,
        # #         callbacks=my_callbacks,
        # # )

        history = model.fit(
            x=train,
            epochs=epoch,
            validation_data=val,
            batch_size=batch_size,
            shuffle=True,
            callbacks=my_callbacks,
        )

        model.save_weights(os.path.join(save_kfold_path, f"KFold{i}", 'last.weights.h5'))
        print("save to >>>>>>>>>> ", os.path.join(save_kfold_path, f"KFold{i}", 'last.weights.h5'))


        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = os.path.join(save_kfold_path, f"KFold{i}", 'history.csv')
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f, index=False)

        model.load_weights(os.path.join(save_kfold_path, f"KFold{i}", "best.weights.h5"))
        val_metrics = model.evaluate(val, batch_size=batch_size)
        # val_metrics = model.evaluate(data_test, labels_one_hot_test, batch_size=batch_size)
        print("Val Loss: ", val_metrics[0])
        print("Val Acc: ", val_metrics[1])

        move_file_kfold(dirs_source=os.path.join(save_path, f"KFold{i}"), dest_folder=dir_image, data_path=train_csv)
        move_file_kfold(dirs_source=os.path.join(save_path, f"KFold{i}"), dest_folder=dir_image, data_path=test_csv)

        

def create_kfold_csv(dir_image, save_path):
    #fix load_images function to get path of image
    _, labels, tmp_paths = load_path_images(dir_images=dir_image, gray=False, dsize=(224, 224))
    labels = np.array(labels)
    tmp_paths = np.array(tmp_paths)
    print(labels.shape, tmp_paths.shape)


    # setup KFOLD
    lb = preprocessing.LabelBinarizer()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10000)
    skf.get_n_splits(tmp_paths, labels)
    print(skf)


    for i, (train_index, test_index) in enumerate(skf.split(tmp_paths, labels)):
        print(labels[train_index].shape)
        labels[test_index]
        train_list = []
        test_list = []
        for index in train_index:
            train_list.append([tmp_paths[index], labels[index]])
        for index in test_index:
            test_list.append([tmp_paths[index], labels[index]])

        os.makedirs(os.path.join(save_path, f"KFold{i}"), exist_ok=True)
        df_train = pd.DataFrame(train_list, columns=[ "path", "label"])
        csv_train = os.path.join(os.path.join(save_path,f"KFold{i}"), "train_path.csv")
        df_train.to_csv(csv_train, index=False)

        df_test = pd.DataFrame(test_list, columns=["path", "label"])
        csv_test = os.path.join(os.path.join(save_path,f"KFold{i}"), "test_path.csv")
        df_test.to_csv(csv_test, index=False)


dir_image = "/work/quang.domanh/datasets/HOVA-1000/HOVA"
save_path = "/work/quang.domanh/ecom-product-categorization/Data_Folds/hova1000"
save_kfold = "/work/quang.domanh/ecom-product-categorization/logs/hova1000"

# model = keras.applications.ResNet50(
#     weights="imagenet",  # Load weights pre-trained on ImageNet.
#     include_top=True,
#     input_shape=(224, 224, 3),
#     pooling = "avg",
# )

base_model = keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
)

training = True
# Freeze the base_model
base_model.trainable = training

model = build_model(base_model, size=(224, 224, 1), num_classs=1000, training=training)

print(model.summary())

# # create_kfold_csv(dir_image, save_path)

train_kfold(model, dir_image, save_path, save_kfold, gray=True, size=(224, 224), batch_size=128)