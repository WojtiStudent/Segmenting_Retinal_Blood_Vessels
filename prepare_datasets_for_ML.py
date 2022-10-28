import utils

import cv2 as cv
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from math import copysign, log10
from collections import Counter


def calc_hu_moments(img):
    moments = cv.moments(img)
    huMoments = cv.HuMoments(moments)

    for i in range(7):
        if huMoments[i] != 0:
            huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))

    return huMoments.flatten()


def set_of_images_preprocessing(img, mask, manual):
    new_mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    new_mask = utils.normalize_image(new_mask)

    new_manual = utils.normalize_image(manual)

    blured = cv.GaussianBlur(img, (5, 5), 50)
    normalized = utils.normalize_image(blured)

    return normalized, new_mask, new_manual


def sliding_window(img, manual, window_size, step):
    features = []
    labels = []

    img_slices = np.lib.stride_tricks.sliding_window_view(img, (window_size, window_size, 3))[::step, ::step]
    manual_slices = np.lib.stride_tricks.sliding_window_view(manual, (window_size, window_size))[::step, ::step]

    max_i, max_j = img_slices.shape[0:2]

    for i in range(max_i):
        for j in range(max_j):
            y = manual_slices[i, j][window_size // 2, window_size // 2]
            img_slice = img_slices[i, j][0]

            hu_moments = calc_hu_moments(img_slice[:, :, 1])[:7]
            red_mean = img_slice[:, :, 0].mean()
            red_var = img_slice[:, :, 0].var()
            green_mean = img_slice[:, :, 1].mean()
            green_var = img_slice[:, :, 1].var()
            blue_mean = img_slice[:, :, 2].mean()
            blue_var = img_slice[:, :, 2].var()

            features.append(np.append(hu_moments, [red_mean, red_var, green_mean, green_var, blue_mean, blue_var]))
            labels.append(y)

    return np.array(features), np.array(labels)

def sliding_window(img, manual, window_size=5, step=5): # window 5x5
    features = []
    labels = []
    
    img_slices = np.lib.stride_tricks.sliding_window_view(img, (window_size, window_size, 3))[::step, ::step]
    manual_slices = np.lib.stride_tricks.sliding_window_view(manual, (window_size, window_size))[::step, ::step]

    max_i, max_j = img_slices.shape[0:2]

    for i in range(max_i):
        for j in range(max_j):
            y = manual_slices[i, j][window_size//2, window_size//2]
            img_slice = img_slices[i, j][0]

            hu_moments = calc_hu_moments(img_slice[:,:,1])[:7]
            red_mean = img_slice[:,:,0].mean()
            red_var = img_slice[:,:,0].var()
            green_mean = img_slice[:,:,1].mean()
            green_var = img_slice[:,:,1].var()
            blue_mean = img_slice[:,:,2].mean()
            blue_var = img_slice[:,:,2].var()

            features.append(np.append(hu_moments, [red_mean, red_var, green_mean, green_var, blue_mean, blue_var]))
            labels.append(y)

       
    return np.array(features), np.array(labels)

def sliding_window_for_dataset(img, manual, mask, window_size=5, step=5): # window 5x5
    features = []
    labels = []
    
    img_slices = np.lib.stride_tricks.sliding_window_view(img, (window_size, window_size, 3))[::step, ::step]
    manual_slices = np.lib.stride_tricks.sliding_window_view(manual, (window_size, window_size))[::step, ::step]

    max_i, max_j = img_slices.shape[0:2]

    for i in range(max_i):
        for j in range(max_j):
            mask_val = mask[window_size//2+i*step, window_size//2+j*step]
            if mask_val == 0:
                continue
                
            y = manual_slices[i, j][window_size//2, window_size//2]
            img_slice = img_slices[i, j][0]

            hu_moments = calc_hu_moments(img_slice[:,:,1])[:7]
            red_mean = img_slice[:,:,0].mean()
            red_var = img_slice[:,:,0].var()
            green_mean = img_slice[:,:,1].mean()
            green_var = img_slice[:,:,1].var()
            blue_mean = img_slice[:,:,2].mean()
            blue_var = img_slice[:,:,2].var()

            features.append(np.append(hu_moments, [red_mean, red_var, green_mean, green_var, blue_mean, blue_var]))
            labels.append(y)

       
    return np.array(features), np.array(labels)

def prepare_datasets(sliding_window_size=5, sliding_window_step=5, verbose=False):
    X_images, y_images = utils.load_all_images()

    X_train_images = X_images[:-6]
    y_train_images = y_images[:-6]

    X_test_images = X_images[-6:]
    y_test_images = y_images[-6:]

    images_groups = [X_train_images, X_test_images]
    manuals_groups = [y_train_images, y_test_images]
    filenames = ["train", "test"]

    curr_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(curr_dir_path, "data_2.0_3x3")
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    for images, manuals, filename in zip(images_groups, manuals_groups, filenames):
        X = []
        y = []

        for (image, mask), manual in tqdm(zip(images, manuals), total=len(images),
                                          desc=f"Processing {filename} images"):
            preprocessed_image, preprocessed_mask, preprocessed_manual = set_of_images_preprocessing(image, mask, manual)
            features, labels = sliding_window_for_dataset(preprocessed_image, preprocessed_manual, preprocessed_mask, window_size=sliding_window_size, step=sliding_window_step)

            X.append(features)
            y.append(labels)

        X_np = np.concatenate(X)
        y_np = np.concatenate(y)
        

        df = pd.DataFrame(X_np, columns=['Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7',
                                      'Red_mean', 'Red_var', 'Green_mean', 'Green_var',
                                      'Blue_mean', 'Blue_var'])
        df["label"] = y_np

        df.to_csv(os.path.join(data_dir_path, f"{filename}.csv"), index=False)


if __name__ == "__main__":
    prepare_datasets(sliding_window_size=5, sliding_window_step=5)

