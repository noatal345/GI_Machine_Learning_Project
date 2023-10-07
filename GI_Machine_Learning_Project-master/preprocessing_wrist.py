import numpy as np
import pandas as pd
import torch
from PIL import Image
import os
from tqdm import tqdm


def generate_data(folder_path, num_of_measurement, shape=(128, 256)):
    # create reference_matrix
    reference_matrix = np.random.randn(num_of_measurement, shape[0] * shape[1])
    # read the csv file
    data = pd.read_csv(folder_path + "dataset.csv")
    # create a new csv file for later use
    new_data = pd.DataFrame(columns=['path', 'label'])
    for i in tqdm(range(len(data))):
        torch.cuda.empty_cache()
        # get the image path from the csv file
        image_path = data.iloc[i, 0]
        # read the png image from the path
        image_path += ".png"
        img = Image.open(image_path)
        # resize the image to shape
        img = img.resize(size=shape, resample=Image.BILINEAR)
        # convert the black and white image to numpy array of shape (shape[0], shape[1])
        img = np.array(img.convert('L'))
        # if cuda is available, move the img to cuda
        if torch.cuda.is_available():
            img = torch.tensor(img, dtype=torch.float32).cuda()
            reference_matrix_cuda = torch.tensor(reference_matrix, dtype=torch.float32).cuda()
            # multiply the reference matrix with the img
            measurements = torch.matmul(reference_matrix_cuda, img.view(shape[0] * shape[1], 1)).cpu().numpy()
        else:
            # multiply the reference matrix with the img
            measurements = np.matmul(reference_matrix, img.reshape(shape[0] * shape[1]))
        # save the measurements to csv file
        pd.DataFrame(measurements).to_csv(folder_path + "measurements/"+str(num_of_measurement) + "_"+str(shape[0]) +
                                          "_" + str(shape[1]) + "/" + str(i) + ".csv", index=False)
        # save the  path and the label to the new general csv file ("new_data")
        new_data = pd.concat([new_data, pd.DataFrame([[folder_path + "measurements/"+str(num_of_measurement) + "_"
                                                       + str(shape[0]) + "_" + str(shape[1]) + "/" + str(i) + ".csv",
                                                       data['label'].iloc[i]]], columns=['path', 'label'])])

    # save the new general csv file
    new_data.to_csv(folder_path + "new_dataset_" + str(num_of_measurement) + "_" + str(shape[0]) + "_"+str(shape[1])
                    + ".csv", index=False)




