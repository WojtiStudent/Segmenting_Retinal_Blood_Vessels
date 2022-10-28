import numpy as np
import skimage
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from tqdm import tqdm

np.random.seed(42)

def load_image(path):
    img = cv.imread(f'{path}')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def load_set_of_images(number, type="dr"):    # type: {"dr", "g", "h"}   
    img = load_image(f'all/images/{number:02d}_{type}.jpg')
    mask = load_image(f'all/mask/{number:02d}_{type}_mask.tif')
    manual = cv.cvtColor(load_image(f'all/manual1/{number:02d}_{type}.tif'), cv.COLOR_RGB2GRAY)
    _, manual = cv.threshold(manual, 128, 255, cv.THRESH_BINARY)

    return img, mask, manual


def load_all_images(n_sets=15):
    n_sets = min(n_sets, 15)
    
    X_images = []
    y_images = []
    
    for img_index in tqdm(range(1, n_sets+1), desc="Loading sets ('dr', 'g', 'h') of images"):
        for img_type in ["dr", "g", "h"]:
            img, mask, manual = load_set_of_images(img_index, img_type)
            X_images.append((img, mask))
            y_images.append(manual)
            
    return X_images, y_images

    
def normalize_image(image):
    image = np.array(image)
    max_val = np.max(image)
    min_val = np.min(image)
    return (image - min_val)/(max_val - min_val)


def print_metrics(result, manual):
    try:
        manual_flattened = manual.flatten()
        result_flattened = result.flatten()
    except:
        manual_flattened = manual
        result_flattened = result
    tn, fp, fn, tp = confusion_matrix(manual_flattened, result_flattened).ravel()
    print("Confusion matrix:\n\n{:^10}|{:^10}\n{}\n{:^10}|{:^10}\n".format(tp,fp,"-"*20,fn,tn))
    print(classification_report_imbalanced(manual_flattened, result_flattened))
    
    
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap="gray")
    plt.show()
    
    
def mark_differences(manual, result):
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    
    marked = np.zeros((*manual.shape,3))
    for i in range(manual.shape[0]):
        for j in range(manual.shape[1]):
            manual_pixel = manual[i,j]
            result_pixel = result[i,j]
            
            if manual_pixel == result_pixel:
                if manual_pixel == 255:
                    marked[i][j] = WHITE
            else:
                if manual_pixel == 255:
                    marked[i][j] = GREEN
                else:
                    marked[i][j] = YELLOW

    return marked.astype('uint8')
    