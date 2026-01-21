"""
    train the UNet model as the classifier for the image segmentation
    parameter settings are saved in the config.yaml file
"""

import numpy as np
import sys
import logging
import os
from os.path import join, exists
import time
import random
from tqdm import tqdm
import glob
import yaml
import torch
from torch import squeeze, from_numpy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms.functional as tf

pwd = os.getcwd()
rootpath_project = os.path.abspath(join(os.getcwd(), "../.."))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.model_training.unet_model import UNet


class TrainingDataset(Dataset):
    def __init__(self, filename_x_predictor, filename_y_label, augment_rotate=False):

        super(TrainingDataset, self).__init__()
        self.augment_rotate = augment_rotate
        self.filename_x_predictor = filename_x_predictor
        self.filename_y_label = filename_y_label

    def __getitem__(self, index):

        predictor = np.load(self.filename_x_predictor[index])
        label = np.load(self.filename_y_label[index])  # label class starts from 0

        if self.augment_rotate:
            angle = random.choices([0.0, 90.0, 180.0, 270.0], k=1)[0]   # randomly select the rotation angle to augment
            predictor = from_numpy(predictor)  # transform as tensor
            predictor = tf.rotate(predictor, angle)
            label = label[np.newaxis, ...]  # add new dimension
            label = from_numpy(label)  # transform as tensor
            label = tf.rotate(label, angle)  # rotate
            label = squeeze(label)  # convert it back to 2d
        else:
            predictor = from_numpy(predictor)  # transform as tensor
            label = from_numpy(label)  # transform as tensor

        return predictor, label

    def __len__(self):
        return len(self.filename_y_label)


def write_list_file(output_filename, list_input):
    with open(output_filename, "w") as f:
        for s in list_input:
            f.write(str(s) + "\n")


# def read_list_file(output_filename):
#     score = []
#     with open(output_filename, "r") as f:
#         for line in f:
#             score.append(line.strip())


def get_training_data_output_path(output_folder, x_training_folder='x_training_topography', y_label_folder='y_label'):
    """
        get the training data output path

        :param output_folder: the output folder name
        :return:
    """
    path_x_output = join(rootpath_project, 'results', 'deep_learning', output_folder, x_training_folder)
    if not exists(path_x_output):
        os.makedirs(path_x_output, exist_ok=True)

    path_y_output = join(rootpath_project, 'results', 'deep_learning', output_folder, y_label_folder)
    if not exists(path_y_output):
        os.makedirs(path_y_output, exist_ok=True)

    return path_x_output, path_y_output


def get_training_testing_filename(config, dir_output, exclude_ccap_flag=False):
    """
    get and save the file name of the training and validation dataset

    :param config:
    :param dir_output:
    :return:
    """

    training_sample_folder = config['training_sample_folder']
    training_sample_x_training_folder = config['training_sample_x_label_folder']
    training_sample_y_label_folder = config['training_sample_y_label_folder']
    path_x_output, path_y_output = get_training_data_output_path(output_folder=training_sample_folder, 
                                                                 x_training_folder=training_sample_x_training_folder,
                                                                 y_label_folder=training_sample_y_label_folder)

    list_filename_x_predictor = glob.glob(join(path_x_output, '*.npy'))
    list_filename_x_predictor.sort()

    list_filename_y_label = glob.glob(join(path_y_output, '*.npy'))
    list_filename_y_label.sort()
    
    if exclude_ccap_flag:
        # remove the CCAP dataset in the training dataset
        mask_ccap_training = [True if 'CCAP' in x else False for x in list_filename_x_predictor]
        mask_ccap_training = np.array(mask_ccap_training)
        
        list_filename_x_predictor = np.array(list_filename_x_predictor)[~mask_ccap_training]
        list_filename_y_label = np.array(list_filename_y_label)[~mask_ccap_training]

    # chip_number = config['chip_number']  # total number of chips
    chip_number = min(config['chip_number'], len(list_filename_x_predictor))  # total number of chips
    split = config['split']  # split the training and validation dataset, split percentage

    # shuffle the training and validation dataset
    shuffle_index = np.random.permutation(chip_number)
    filename_predictor = list(np.array(list_filename_x_predictor)[shuffle_index])
    filename_label = list(np.array(list_filename_y_label)[shuffle_index])

    # split the training and validation dataset
    training_split_count = int(chip_number * split)
    tmp = np.arange(chip_number)
    training_split_index = tmp[0: training_split_count]
    validation_split_index = tmp[training_split_count: len(tmp)]

    # file name of the training predictor variables and labels
    filename_training_predictor = list(np.array(filename_predictor)[training_split_index])
    filename_training_label = list(np.array(filename_label)[training_split_index])

    # file name of the validation predictor variables and labels
    filename_validation_predictor = list(np.array(filename_predictor)[validation_split_index])
    filename_validation_label = list(np.array(filename_label)[validation_split_index])

    write_list_file(join(dir_output, 'filename_training_predictor.txt'), filename_training_predictor)
    write_list_file(join(dir_output, 'filename_training_label.txt'), filename_training_label)
    write_list_file(join(dir_output, 'filename_validation_predictor.txt'), filename_validation_predictor)
    write_list_file(join(dir_output, 'filename_validation_label.txt'), filename_validation_label)

    return filename_training_predictor, filename_training_label, filename_validation_predictor, filename_validation_label


def get_loss_function(config, device):
    """
    get the loss function based on the configuration file

    :param config:
    :param device:
    :return:
    """
    loss_function_flag = config['loss_function_flag']

    if loss_function_flag == 'cross_entropy':
        loss_function = nn.CrossEntropyLoss()
        logging.info('Standard cross entropy loss used')
    elif loss_function_flag == 'focal_loss':
        from pythoncode.model_training.focal_loss import focal_loss

        loss_function = focal_loss(device=device)
        logging.info('Loss: focal loss')
    else:
        loss_function = nn.MSELoss()
        logging.info('MSE loss is used for continuous variable')

    return loss_function, loss_function_flag


def get_unet_model(config, device, dir_output):
    """
    get the U-Net model based on the configuration file, including two conditions
    (1) Previous epochs of trained model exist, then load the latest model
    (2) No previous epochs, start from the beginning

    :param config:
    :param device:
    :param dir_output:
    :return:
    """

    in_channels = config['in_channels']
    num_classes = config['num_classes']
    learning_rate = config['learning_rate']

    unet_model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)

    # ref to explain the zero_grad: https://zhuanlan.zhihu.com/p/115575367
    optimizer = torch.optim.Adam(params=unet_model.parameters(), lr=learning_rate)

    # load the existing model, or start from the beginning
    list_existing_model = glob.glob(join(dir_output, 'model_output', '*pth'))
    list_existing_model.sort()

    if len(list_existing_model) > 0:
        print(os.path.split(list_existing_model[-1])[-1][-6:-4])
        existing_epoch = int(os.path.split(list_existing_model[-1])[-1][-6:-4])

        checkpoint = torch.load(list_existing_model[-1])
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
    else:
        existing_epoch = 0

    return unet_model, optimizer, existing_epoch


def load_training_dataset(filename_training_predictor, filename_training_label, config):
    """
        load the training dataset

    :param filename_training_predictor:
    :param filename_training_label:
    :param config:
    :return:
    """

    dataset = TrainingDataset(filename_training_predictor, filename_training_label, augment_rotate=config['augment_training'])

    batch_size = config['batch_size']
    shuffle = config['shuffle_training']
    num_workers = config['num_workers']
    prefetch_factor = config['prefetch_factor']

    # ref about the DataLoader parameter meaning : https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    training_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=False, prefetch_factor=prefetch_factor, persistent_workers=True)

    return training_dataset


def load_validation_dataset(filename_validation_predictor, filename_validation_label, config):
    """
        load the training dataset

    :param filename_training_predictor:
    :param filename_training_label:
    :param config:
    :return:
    """

    dataset = TrainingDataset(filename_validation_predictor, filename_validation_label,
                              augment_rotate=config['augment_test'])

    batch_size = config['batch_size']
    shuffle = config['shuffle_test']
    num_workers = config['num_workers']
    prefetch_factor = config['prefetch_factor']

    # ref about the DataLoader parameter meaning : https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    training_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=False, prefetch_factor=prefetch_factor, persistent_workers=True)

    return training_dataset


def model_train(training_dataset, device, optimizer, unet_model, loss_function_flag, loss_function, logging):
    """
    train the UNet model

    :param training_dataset:
    :param device:
    :param optimizer:
    :param unet_model:
    :param loss_function:
    :param logging:
    :return:
    """

    loop = tqdm(training_dataset)  # progress

    list_train_loss = []
    list_train_accuracy = []

    for image, label in loop:
        image = image.to(device, dtype=torch.float)

        if (loss_function_flag == 'cross_entropy') | (loss_function_flag == 'focal_loss'):
            # classification target

            label = label.to(device, dtype=torch.long)  # long is for computing the loss

            unet_model.train()

            image_pred = unet_model(image)  # predicted tensor
            train_mse = loss_function(image_pred, label)   # calculate the loss

            optimizer.zero_grad()
            train_mse.backward()
            optimizer.step()

            # calculate the accuracy. This is for classifier, so we need to convert the predicted probability to the label to get the accuracy
            label_pred = torch.argmax(torch.softmax(image_pred, dim=1), dim=1)  # predicted label using the softmax
            train_accuracy = ((label_pred == label).sum() / torch.numel(label_pred))    # calculate the accuracy, i.e., overall accuracy

            list_train_loss.append(train_mse.item())
            list_train_accuracy.append(train_accuracy.item())

            logging.info('Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(train_mse.item(), train_accuracy.item()))

        else:
            # regression target
            label = label.to(device, dtype=torch.float)  # float type, for continuous variable

            unet_model.train()

            image_pred = unet_model(image)  # predicted tensor
            img_pred_squeeze = image_pred.clone()   # clone the tensor
            img_pred_squeeze = img_pred_squeeze.squeeze(dim=1)  # remove dimensions of size 1 from the predicted tensor to match the label
            train_mse = loss_function(img_pred_squeeze, label) # calculate the loss (mean square error) between the predicted tensor and the label

            optimizer.zero_grad()
            train_mse.backward()
            optimizer.step()

            # calculate the mean absolute error
            train_mae = nn.L1Loss()
            train_mae = train_mae(img_pred_squeeze, label)

            # append the loss and accuracy to the output list
            list_train_loss.append(train_mse.item())
            list_train_accuracy.append(train_mae.item())

            logging.info('Train MSE: {:.4f}, Train L1 (MAE): {:.4f}'.format(train_mse.item(), train_mae.item()))

    return list_train_loss, list_train_accuracy


def validation(validation_dataset, device, unet_model, loss_function_flag, loss_function, logging):
    """
    validate the UNet model on the testing dataset

    :param validation_dataset:
    :param device:
    :param unet_model:
    :param loss_function:
    :param logging:
    :return:
    """
    loop = tqdm(validation_dataset)  # progress

    list_val_loss = []
    list_val_accuracy = []

    for image_val, label_val in loop:
        image_val = image_val.to(device, dtype=torch.float)
        label_val = label_val.to(device, dtype=torch.long)  # long is for computing the loss

        unet_model.eval()

        image_pred = unet_model(image_val)  # predicted image

        if (loss_function_flag == 'cross_entropy') | (loss_function_flag == 'focal_loss'):
            # classification target

            validation_loss = loss_function(image_pred, label_val)  # loss value

            label_pred = torch.argmax(torch.softmax(image_pred, dim=1), dim=1)  # label
            validation_accuracy = ((label_pred == label_val).sum() / torch.numel(label_pred))

            list_val_loss.append(validation_loss.item())
            list_val_accuracy.append(validation_accuracy.item())

            logging.info('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(validation_loss.item(), validation_accuracy.item()))

        else:
            # regression target
            img_pred_squeeze = image_pred.clone()  # clone the tensor
            img_pred_squeeze = img_pred_squeeze.squeeze(dim=1)  # remove dimensions of size 1 from the predicted tensor to match the label

            validation_mse = loss_function(img_pred_squeeze, label_val)   # calculate the loss (mean square error) between the predicted tensor and the label

            # calculate the mean absolute error
            l1_loss = nn.L1Loss()
            validation_mae = l1_loss(img_pred_squeeze, label_val)

            list_val_loss.append(validation_mse.item())
            list_val_accuracy.append(validation_mae.item())

            logging.info('Validation MSE: {:.4f}, Validation L1 (MAE): {:.4f}'.format(validation_mse.item(), validation_mae.item()))

    return list_val_loss, list_val_accuracy


# def main():
if __name__ == '__main__':

    with open(join(pwd, 'config.yaml')) as file:
        config = yaml.full_load(file)

    # with open(join(r'K:\CSM_project\results\deep_learning\v_train_classifier_v1\model_training_output\v_train_classifier_v1_config.yaml')) as file:
    #     config = yaml.full_load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    ##
    training_version = config['training_version']
    path_unet_folder = join(rootpath_project, 'results', 'deep_learning')

    dir_output = join(path_unet_folder, training_version, 'model_training_output')
    if not os.path.exists(dir_output):
        os.makedirs(dir_output, exist_ok=True)

    # save the configuration file
    with open(join(dir_output, '{}_config.yaml'.format(training_version)), 'w') as file:
        yaml.dump(config, file)

    logging.basicConfig(filename=join(dir_output, '{}_training.log'.format(training_version)),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logging.info('training version: {}'.format(training_version))
    logging.info('training sample folder {}'.format(config['training_sample_folder']))
    logging.info('device {}'.format(device))
    ##
    # get the training and validation dataset file name
    (filename_training_predictor, filename_training_label,
     filename_validation_predictor, filename_validation_label) = get_training_testing_filename(config, 
                                                                                               dir_output,
                                                                                               exclude_ccap_flag=config['exclude_ccap'])
    ##
    training_dataset = load_training_dataset(filename_training_predictor, filename_training_label, config)
    validation_dataset = load_validation_dataset(filename_validation_predictor, filename_validation_label, config)

    logging.info('prepare training and validation dataset finished, total chip size: {}, split percentage: {}'.
                 format(len(filename_training_predictor) + len(filename_validation_predictor), config['split']))

    ##
    # define the loss function
    loss_function, loss_function_flag= get_loss_function(config, device)

    ##
    unet_model, optimizer, existing_epoch = get_unet_model(config, device, dir_output)
    logging.info('UNet model initialization finished')

    ##
    epochs = config['epochs']
    for epoch in range(existing_epoch, epochs):

        logging.info('epoch {:03d} start'.format(epoch + 1))
        start_epoch = time.perf_counter()
        print('epoch:{}'.format(epoch + 1))

        list_train_loss, list_train_accuracy = model_train(training_dataset, device, optimizer, unet_model,
                                                           loss_function_flag, loss_function, logging)

        list_test_loss, list_test_accuracy = validation(validation_dataset, device, unet_model,
                                                        loss_function_flag, loss_function, logging)

        dir_model = join(dir_output, 'model_output')
        if not os.path.exists(dir_model):
            os.makedirs(dir_model, exist_ok=True)

        # model save ref: https://zhuanlan.zhihu.com/p/422797058
        torch.save({
            'model_state_dict': unet_model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'list_train_loss': list_train_loss,
            'list_train_accuracy': list_train_accuracy,
            'list_test_loss': list_test_loss,
            'list_test_accuracy': list_test_accuracy,   # typo here, should be list_test_accuracy
        }, join(dir_model, 'unet_model_record_{:02d}.pth'.format(epoch + 1)))

        end_epoch = time.perf_counter()
        running_time = end_epoch - start_epoch

        if (loss_function_flag == 'cross_entropy') | (loss_function_flag == 'focal_loss'):
            print('epoch {:03d} finished in {:.2f} minutes, training loss {:.3f}, training accuracy {:.3f}, '
                  'test loss {:.3f}, test accuracy {:.3f}'.
                  format(epoch + 1, running_time / 60, np.nanmean(list_train_loss), np.nanmean(list_train_accuracy),
                         np.nanmean(list_test_loss), np.nanmean(list_test_accuracy)))

            logging.info('epoch {:03d} finished in {:.2f} minutes, training loss {:.3f}, training accuracy {:.3f}, '
                         'test loss {:.3f}, test accuracy {:.3f}'.
                         format(epoch + 1, running_time / 60, np.nanmean(list_train_loss), np.nanmean(list_train_accuracy),
                                np.nanmean(list_test_loss), np.nanmean(list_test_accuracy)))
        else:
            print('epoch {:03d} finished in {:.2f} minutes, training MSE {:.3f}, training MAE {:.3f}, '
                  'test MSE {:.3f}, test MAE {:.3f}'.
                  format(epoch + 1, running_time / 60, np.nanmean(list_train_loss), np.nanmean(list_train_accuracy),
                         np.nanmean(list_test_loss), np.nanmean(list_test_accuracy)))

            logging.info('epoch {:03d} finished in {:.2f} minutes, training MSE {:.3f}, training MAE {:.3f}, '
                         'test MSE {:.3f}, test MAE {:.3f}'.
                         format(epoch + 1, running_time / 60, np.nanmean(list_train_loss), np.nanmean(list_train_accuracy),
                                np.nanmean(list_test_loss), np.nanmean(list_test_accuracy)))
