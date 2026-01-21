"""
predict the ISP with the trained UNet model
"""

from osgeo import gdal, gdalconst, gdal_array
import os
from os.path import join
import numpy as np
import sys
import logging
import yaml
import torch
from torch import from_numpy

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.model_training.unet_model import UNet
from pythoncode.model_training.utils_deep_learning import (read_cold_variable,
                                                           predictor_normalize,
                                                           read_topography_data,
                                                           topography_normalize,
                                                           get_proj_info,
                                                           add_pyramids_color_in_nlcd_isp_tif, )


def unet_model_load(config, device='cpu', epoch=1, rootpath_save=None):
    """
    load the UNet model

    :param config:
    :param device:
    :param epoch:
    :param rootpath_save:
    :return:
    """

    in_channels = config['in_channels']  # the dimension of predictor variables
    num_classes = config['num_classes']  # the number of classes, 1 for regression, and 101 for ISP classification

    unet_model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)

    filename_model = join(rootpath_save, 'model_training_output', 'model_output', 'unet_model_record_{:02d}.pth'.format(epoch))
    print(filename_model)
    checkpoint = torch.load(filename_model, map_location=torch.device(device), weights_only=True)

    unet_model.load_state_dict(checkpoint['model_state_dict'])

    unet_model.eval()

    return unet_model


def predict_isp_classification(predictor, device, unet_model):
    """
    predict isp using the UNet classification model

    :param predictor: the input predictor variables
    :param device:
    :param unet_model:
    :return:
    """
    predictor = from_numpy(predictor)
    tensor_image = torch.unsqueeze(predictor, dim=0)

    tensor_image = tensor_image.to(device, dtype=torch.float)

    image_pred = unet_model(tensor_image)  # predicted image
    label_pred = torch.argmax(torch.softmax(image_pred, dim=1), dim=1)  # get the predicted label using softmax

    numpy_pred = label_pred.cpu().detach().numpy()
    numpy_pred = numpy_pred.astype(np.float32)
    numpy_pred = numpy_pred[0, :, :]

    return numpy_pred


def predict_isp_regression(predictor, device, unet_model):
    """
    predict isp using the UNet regression model
    The softmax function is not used in the regression model

    :param predictor: the input predictor variables
    :param device:
    :param unet_model:
    :return:
    """
    predictor = from_numpy(predictor)
    tensor_image = torch.unsqueeze(predictor, dim=0)

    tensor_image = tensor_image.to(device, dtype=torch.float)

    image_pred = unet_model(tensor_image)  # predicted image
    image_pred = image_pred.squeeze()

    numpy_pred = image_pred.cpu().detach().numpy()

    return numpy_pred


def mosaic_prediction(predictor, device, unet_model, prediction_type='classification'):
    """
        predict isp for each Landsat ARD tile

        HPC cannot directly predict the 5000 by 5000 image due to the RAM limitation, so separate them first and
        them combine the prediction results

        For a 5000 by 5000 image, it is separated into 4 2500 by 2500 images, and then predict the 4 images separately
        The required RAM/memory about 24G, recommend to apply 30GB

        :return numpy_pred_mosaic: the predicted (5000, 5000) size ISP image
    """

    nrow, ncol = np.shape(predictor)[1], np.shape(predictor)[2]
    assert nrow % 2 == 0
    assert ncol % 2 == 0

    seg_nrow = nrow // 2
    seg_ncol = ncol // 2

    numpy_pred_mosaic = np.zeros((nrow, ncol), dtype=np.float32)

    for p in range(0, nrow, seg_nrow):
        for q in range(0, ncol, seg_ncol):

            if prediction_type == 'classification':
                numpy_pred_tmp = predict_isp_classification(predictor[:, p:p + seg_nrow, q:q + seg_ncol], device, unet_model)
            else:
                numpy_pred_tmp = predict_isp_regression(predictor[:, p:p + seg_nrow, q:q + seg_ncol], device, unet_model)

            numpy_pred_mosaic[p:p + seg_nrow, q:q + seg_ncol] = numpy_pred_tmp

    return numpy_pred_mosaic


def predict_isp_output(output_path, tile_name, year, numpy_pred, src_geotrans, src_proj,
                       gdal_type=gdalconst.GDT_Float32,
                       filename_prefix='unet_classifier'):
    """
    output the isp

    :param output_path:
    :param tile_name:
    :param year:
    :param numpy_pred:
    :param src_geotrans:
    :param src_proj:
    :return:
    """

    output_filename = join(output_path, '{}_{}_{}_isp.tif'.format(filename_prefix, tile_name, year))

    n_row, n_col = numpy_pred.shape[0], numpy_pred.shape[1]
    tif_output = gdal.GetDriverByName('GTiff').Create(output_filename, n_col, n_row, 1, gdal_type, options=['COMPRESS=LZW'])
    tif_output.SetGeoTransform(src_geotrans)
    tif_output.SetProjection(src_proj)

    Band = tif_output.GetRasterBand(1)
    Band.WriteArray(numpy_pred)

    tif_output = None
    del tif_output

    return output_filename


def pipe_line_unet_prediction(tile_name, list_year, task_type_flag, training_version, epoch,
                              central_reflectance_flag,
                              output_folder_name, rootpath_project_folder=None,
                              norm_boundary_folder='maximum_minimum_ref'):
    """pipeline to predict the ISP using the UNet model

    Args:
        tile_name (_type_): _description_
        list_year (_type_): _description_
        task_type_flag (_type_): regression or classification
        training_version (_type_): training version of UNet model
        epoch (_type_): epoch number
        central_reflectance_flag (_type_): central reflectance flag, 'change' or 'stable'
        output_folder_name (_type_): prediction output folder name: predict_isp_change or predict_is_mask
    """

    dir_root_folder = join(rootpath_project, 'results', 'deep_learning', training_version)  # the root folder of the training model

    # load the training configuration
    with open(join(dir_root_folder, 'model_training_output', '{}_config.yaml'.format(training_version))) as file_config:
        config = yaml.full_load(file_config)

    if rootpath_project_folder is None:
        output_path = join(dir_root_folder, output_folder_name, tile_name)
    else:
        output_path = join(rootpath_project_folder, 'results', 'deep_learning', training_version, output_folder_name, tile_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    logging.basicConfig(filename=join(output_path, '{}_isp_prediction.log'.format(tile_name)),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info('device {}'.format(device))

    unet_model = unet_model_load(config=config, device=device, epoch=epoch, rootpath_save=dir_root_folder)    
    logging.info(f'{task_type_flag} unet model epoch {epoch} load finished')

    for i_year in range(0, len(list_year)):
        # for i_year in range(0, 1):
        year = list_year[i_year]
        print('predicting isp for tile {} year {}'.format(tile_name, year))
        logging.info('isp prediction for tile {} year {}'.format(tile_name, year))

        if config['in_channels'] == 35:
            # means only spectral bands are used
            predictor = read_cold_variable(predictor_variable_folder='predictor_variable',
                                           tile_name=tile_name,
                                           year=year,
                                           central_reflectance_flag=central_reflectance_flag,
                                           rootpath_project_folder=rootpath_project_folder)

            predictor = predictor_normalize(predictor, norm_boundary_folder=norm_boundary_folder)
        else:
            # spectral features and topographic features are used
            predictor = read_cold_variable(predictor_variable_folder='predictor_variable',
                                           tile_name=tile_name,
                                           year=year,
                                           central_reflectance_flag=central_reflectance_flag,
                                           rootpath_project_folder=rootpath_project_folder)

            # normalize the predictor variables
            predictor = predictor_normalize(predictor, norm_boundary_folder=norm_boundary_folder)

            # read the topography data
            img_dem, img_slope, img_aspect = read_topography_data(tile_name)
            # normalize the topography data
            img_dem, img_slope, img_aspect = topography_normalize(img_dem, img_slope, img_aspect, norm_boundary_folder=norm_boundary_folder)

            predictor = np.concatenate((predictor, img_dem, img_slope, img_aspect), axis=0)

        logging.info('training sample from {}'.format(config['training_sample_folder']))
        logging.info('central reflectance flag is {}'.format(central_reflectance_flag))

        src_proj, src_geotrans = get_proj_info(tile_name)

        ##
        if task_type_flag == 'classification':
            # classification task
            numpy_pred_mosaic = mosaic_prediction(predictor, device, unet_model, prediction_type='classification')
            logging.info('isp prediction finished')

            numpy_pred_mosaic[np.isnan(numpy_pred_mosaic)] = 255
            output_filename = predict_isp_output(output_path, tile_name, year, numpy_pred_mosaic, src_geotrans, src_proj,
                                                 filename_prefix='unet_classifier',
                                                 gdal_type=gdalconst.GDT_Byte)
            logging.info('output the isp map {}'.format(output_filename))

            add_pyramids_color_in_nlcd_isp_tif(output_filename)
        else:
            # regression task
            numpy_pred_mosaic = mosaic_prediction(predictor, device, unet_model, prediction_type='regression')
            logging.info('isp prediction finished')

            output_filename = predict_isp_output(output_path, tile_name, year, numpy_pred_mosaic,
                                                 src_geotrans, src_proj,
                                                 filename_prefix='unet_regressor_original',
                                                 gdal_type=gdalconst.GDT_Float32)
            logging.info('output the isp map {}'.format(output_filename))
            logging.info('save the original isp prediction image for tile {} year {}'.format(tile_name, year))

            numpy_pred_mosaic_cutoff = numpy_pred_mosaic.copy()
            numpy_pred_mosaic_cutoff[numpy_pred_mosaic_cutoff > 100] = 100
            numpy_pred_mosaic_cutoff[numpy_pred_mosaic_cutoff < 0] = 0

            numpy_pred_mosaic_cutoff_round = np.round(numpy_pred_mosaic_cutoff)
            numpy_pred_mosaic_cutoff_round[np.isnan(numpy_pred_mosaic_cutoff_round)] = 255
            numpy_pred_mosaic_cutoff_round = numpy_pred_mosaic_cutoff_round.astype(np.uint8)

            output_filename = predict_isp_output(output_path, tile_name, year, numpy_pred_mosaic_cutoff_round,
                                                 src_geotrans, src_proj,
                                                 filename_prefix='unet_regressor_round',
                                                 gdal_type=gdalconst.GDT_Byte)
            logging.info('output the isp map {}'.format(output_filename))

            add_pyramids_color_in_nlcd_isp_tif(output_filename)


def mask_unet_regressor_isp_with_binary_mask(tile_name, year,
                                             training_version_isp_regression, isp_regression_folder, filename_prefix_isp_regression,
                                             training_version_is_mask, is_mask_folder, filename_prefix_is_mask,
                                             filename_prefix_mask_isp_output,
                                             rootpath_project_folder=None
                                             ):
    """mask the ISP from U-Net regressor with the binary IS mask from U-Net binary classifier

    Args:
        tile_name (_type_): _description_
        year (_type_): _description_
        training_version_isp_regression (_type_): the training version of the U-Net regressor
        isp_regression_folder (str): the output folder of the U-Net regressor, e.g., 'predict_isp_change'
        filename_prefix_isp_regression (_type_): the prefix of the ISP file from U-Net regressor
        training_version_is_mask (_type_): the training version of the U-Net binary classifier
        is_mask_folder (str): the output folder of the U-Net binary classifier, e.g., 'predict_is_mask'
        filename_prefix_is_mask (_type_): the prefix of the IS mask file from U-Net binary classifier

        filename_prefix_mask_isp_output (_type_): the prefix of the masked ISP file
    """

    if rootpath_project_folder is None:
        rootpath_project_folder = rootpath_project

    filename_predicted_isp = join(rootpath_project_folder, 'results', 'deep_learning', training_version_isp_regression,
                                  isp_regression_folder, tile_name,
                                  '{}_{}_{}_isp.tif'.format(filename_prefix_isp_regression, tile_name, year))

    img_predicted_isp = gdal_array.LoadFile(filename_predicted_isp)

    filename_mask = join(rootpath_project_folder, 'results', 'deep_learning', training_version_is_mask,
                         is_mask_folder, tile_name,
                         '{}_{}_{}_isp.tif'.format(filename_prefix_is_mask, tile_name, year))
    img_mask = gdal_array.LoadFile(filename_mask)

    img_predicted_isp_masked = img_predicted_isp * img_mask

    src_proj, src_geotrans = get_proj_info(tile_name)

    output_path = join(rootpath_project_folder, 'results', 'deep_learning', training_version_isp_regression, isp_regression_folder, tile_name)

    output_filename = predict_isp_output(output_path, tile_name, year, img_predicted_isp_masked,
                                         src_geotrans, src_proj,
                                         filename_prefix=filename_prefix_mask_isp_output,
                                         gdal_type=gdalconst.GDT_Byte)

    add_pyramids_color_in_nlcd_isp_tif(output_filename)

    return output_filename


# def main():
if __name__ == "__main__":

    list_year = np.array([2013, 2014, 2017, 2018])
    list_tile_name = ['h027v008', 'h028v008', 'h027v009']

    # list_year = np.arange(1985, 2022, 1)
    # from conus_isa_production.utils_get_conus_tile_name import get_conus_tile_name
    # list_tile_name = get_conus_tile_name()

    # list_tile_name = os.listdir(join(output_rootpath, 'data', 'predictor_variable'))

    training_version = 'regressor_v5_conus_topography'
    task_type_flag = 'regression'  # 'classification' or 'regression'

    output_folder_name = 'predict_isp_change'  # 'predict_isp_change' or 'predict_is_mask'

    central_reflectance_flag = 'change'
    norm_boundary_folder = 'maximum_minimum_ref_conus'
    epoch_reg = 98
    epoch_cls = 100

    training_version_isp_regression = 'regressor_v6_conus_topography'  # version of the training model, e.g., 'classifier_v1', 'regressor_v1'
    isp_regression_folder = 'epoch98_predict_isp_change'
    filename_prefix_isp_regression = 'unet_regressor_round'

    training_version_is_mask = 'binary_classification_v6_conus_topography'  # 'binary_classification_conus_v1', 'binary_classification_v1'
    is_mask_folder = 'epoch100_predict_is_mask'
    filename_prefix_is_mask = 'unet_classifier'

    filename_prefix_mask_isp_output = 'unet_regressor_round_masked'

    output_rootpath = '/gpfs/sharedfs1/zhulab/Falu/CSM_project/'

    for i_tile, tile_name in enumerate(list_tile_name):
        for i_year, year in enumerate(list_year):

            print('predicting isp for tile {}'.format(tile_name))

            pipe_line_unet_prediction(tile_name=tile_name,
                                      list_year=list_year,
                                      task_type_flag='classification',
                                      training_version=training_version_is_mask,
                                      epoch=epoch_cls,
                                      central_reflectance_flag=central_reflectance_flag,
                                      output_folder_name=is_mask_folder,
                                      rootpath_project_folder=output_rootpath,
                                      norm_boundary_folder=norm_boundary_folder)

            pipe_line_unet_prediction(tile_name=tile_name,
                                      list_year=list_year,
                                      task_type_flag='regression',
                                      training_version=training_version_isp_regression,
                                      epoch=epoch_reg,
                                      central_reflectance_flag=central_reflectance_flag,
                                      output_folder_name=isp_regression_folder,
                                      rootpath_project_folder=output_rootpath,
                                      norm_boundary_folder=norm_boundary_folder)

            # mask the original regression ISP output with the binary classification mask
            output_filename = mask_unet_regressor_isp_with_binary_mask(tile_name=tile_name, year=year,
                                                                       training_version_isp_regression=training_version_isp_regression,
                                                                       isp_regression_folder=isp_regression_folder,
                                                                       filename_prefix_isp_regression=filename_prefix_isp_regression,
                                                                       training_version_is_mask=training_version_is_mask,
                                                                       is_mask_folder=is_mask_folder,
                                                                       filename_prefix_is_mask=filename_prefix_is_mask,
                                                                       filename_prefix_mask_isp_output=filename_prefix_mask_isp_output,
                                                                       rootpath_project_folder=output_rootpath,
                                                                       )
            print(output_filename)


# if __name__ == "__main__":
#     main()
