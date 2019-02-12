# import os
from os import path, mkdir
from pixel_decoder.utils import cache_stats, batch_data_generator, val_data_generator

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
from sklearn.model_selection import KFold
# import cv2
from keras.optimizers import SGD, Adam
from keras.models import model_from_json
from keras import metrics
from keras.callbacks import ModelCheckpoint
from pixel_decoder.loss import dice_coef, dice_logloss2, dice_logloss3, dice_coef_rounded, dice_logloss
from pixel_decoder.resnet_unet import get_resnet_unet
import keras.backend as K
import json
from keras.callbacks import LambdaCallback



#trying to add pyveda for generator
import pyveda as pv

def train(batch_size, imgs_folder, masks_folder, model_id, origin_shape_no,
          border_no, number_of_epochs, veda_data, models_folder=False, classes =1, channel_no=3,
          return_model=True):
    origin_shape = (int(origin_shape_no), int(origin_shape_no))
    border = (int(border_no), int(border_no))
    input_shape = origin_shape
    all_files, all_masks = imgs_folder, masks_folder
    means, stds = cache_stats(imgs_folder)
    if model_id == 'resnet_unet':
        #model = get_resnet_unet(input_shape, channel_no)
        model = get_resnet_unet(input_shape, channel_no, classes)
    else:
        print('No model loaded!')

    if models_folder is False:
        models_folder='model'
        mkdir(models_folder)

    else:
        if not path.isdir(models_folder):
            mkdir(models_folder)
    if not path.exists('model_stats'):
        mkdir('model_stats')
    kf = KFold(n_splits=4, shuffle=True, random_state=1)
    for all_train_idx, all_val_idx in kf.split(all_files):
    #     train_idx = []
    #     val_idx = []
    #
    #     for i in all_train_idx:
    #         train_idx.append(i)
    #     for i in all_val_idx:
    #         val_idx.append(i)
    #
    #     validation_steps = int(len(val_idx) / batch_size)
    #     steps_per_epoch = int(len(train_idx) / batch_size)

        validation_steps = int(len(veda_data.validate)/batch_size)
        steps_per_epoch = int(len(veda_data.train)/batch_size)
        if validation_steps == 0 or steps_per_epoch == 0:
            continue
        print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

        np.random.seed(11)
        random.seed(11)
        tf.set_random_seed(11)
        print(model.summary())
        # batch_data_generat = batch_data_generator(train_idx, batch_size, means, stds, imgs_folder, masks_folder, models_folder, channel_no, border_no, origin_shape_no)
        # val_data_generat = val_data_generator(val_idx, batch_size, validation_steps, means, stds, imgs_folder, masks_folder, models_folder, channel_no, border_no, origin_shape_no)

        batch_data_generat = veda_data.train.batch_generator(batch_size=batch_size, channels_last=True)
        for x,y in batch_data_generat:
            print('train', x.shape, y.shape)
        val_data_generat = veda_data.test.batch_generator(batch_size=batch_size, channels_last=True)
        for x,y in val_data_generat:
            print('validation', x.shape, y.shape)

        print("about to compile")
        model.compile(loss=dice_logloss3,
                    optimizer=SGD(lr=5e-2, decay=1e-6, momentum=0.9, nesterov=True),
                    metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])

        print("model complied")

        model_checkpoint = ModelCheckpoint(path.join(models_folder, '{}_weights.h5'.format(model_id)), monitor='val_dice_coef_rounded',
                                         save_best_only=True, save_weights_only=False, mode='max')
        model_1=model.fit_generator(generator=batch_data_generat,
                            epochs=number_of_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
                            validation_data=val_data_generat,
                            validation_steps=validation_steps,
                            callbacks=[model_checkpoint], workers=0)

        # for l in model.layers:
        #     l.trainable = True
        # model.compile(loss=dice_logloss3,
        #             optimizer=Adam(lr=1e-3),
        #             metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        #
        # model_2=model.fit_generator(generator=batch_data_generat,
        #                     epochs=number_of_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
        #                     validation_data=val_data_generat,
        #                      validation_steps=validation_steps,
        #                     callbacks=[model_checkpoint], use_multiprocessing=True)
        # model_2_stats=model_2.history['loss']
        # model_num=2
        # numpy_loss_history = np.array(model_2_stats)
        #
        # model.optimizer = Adam(lr=2e-4)
        # model.fit_generator(generator=batch_data_generat,
        #                     epochs=number_of_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
        #                     validation_data=val_data_generat,
        #                     validation_steps=validation_steps,
        #                     callbacks=[model_checkpoint], use_multiprocessing=True)
        #
        # np.random.seed(22)
        # random.seed(22)
        # tf.set_random_seed(22)
        # model.load_weights(path.join(models_folder, '{}_weights2.h5'.format(model_id)))
        # model.compile(loss=dice_logloss,
        #             optimizer=Adam(lr=5e-4),
        #             metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        # model_checkpoint2 =  ModelCheckpoint(path.join(models_folder, '{}_weights.h5'.format(model_id)), monitor='val_dice_coef_rounded',
        #                                  save_best_only=True, save_weights_only=False, mode='max')
        # # model.fit_generator(generator=batch_data_generat,
        # #                     epochs=number_of_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
        # #                     validation_data=val_data_generat,
        # #                     validation_steps=validation_steps,
        # #                     callbacks=[model_checkpoint2])
        # # optimizer=Adam(lr=1e-5)
        # # model.fit_generator(generator=batch_data_generat,
        # #                     epochs=number_of_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
        # #                     validation_data=val_data_generat,
        # #                     validation_steps=validation_steps,
        # #                     callbacks=[model_checkpoint2])
        # # model_2_stats = model.history()
        # # np.random.seed(33)
        # # random.seed(33)
        # # tf.set_random_seed(33)
        # # model.load_weights(path.join(models_folder, '{}_weights2.h5'.format(model_id)))
        # # model.compile(loss=dice_logloss2,
        # #             optimizer=Adam(lr=5e-5),
        # #             metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        # # model_checkpoint3 = ModelCheckpoint(path.join(models_folder, '{}_weights.h5'.format(model_id)), monitor='val_dice_coef_rounded',
        # #                                  save_best_only=True, save_weights_only=False, mode='max')
        # # model.fit_generator(generator=batch_data_generat,
        # #                     epochs=number_of_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
        # #                     validation_data=val_data_generat,
        # #                     validation_steps=validation_steps,
        # #                     callbacks=[model_checkpoint3])
        # # model_3_stats = model.history()
        # # np.random.seed(44)
        # # random.seed(44)
        # # tf.set_random_seed(44)
        # # model.load_weights(path.join(models_folder, '{}_weights3.h5'.format(model_id)))
        # # model.compile(loss=dice_logloss3,
        # #             optimizer=Adam(lr=2e-5),
        # #             metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
        # # model_checkpoint4 = ModelCheckpoint(path.join(models_folder, '{}_weights.h5'.format(model_id)), monitor='val_dice_coef_rounded',
        # #                                  save_best_only=True, save_weights_only=False, mode='max')
        # # model.fit_generator(generator=batch_data_generat,
        # #                     epochs=number_of_epochs, steps_per_epoch=steps_per_epoch, verbose=2,
        # #                     validation_data=val_data_generat,
        # #                     validation_steps=validation_steps,
        # #                     callbacks=[model_checkpoint4])
        # # model_4_stats = model.history()
        # # json_log = open('loss_log.json', mode='wt', buffering=1)
        # # json_logging_callback = LambdaCallback(
        # #     on_epoch_end=lambda epoch, logs: json_log.write(
        # #         json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
        # #     on_train_end=lambda logs: json_log.close()
        # # )

        K.clear_session()
