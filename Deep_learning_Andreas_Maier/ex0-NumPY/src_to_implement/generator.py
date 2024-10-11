import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
import random
import copy


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    labels_dict = {}
    images = []
    labels = []
    current_index: int

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.current_index = 0
        self.epoch = 0

        temp_images = []
        temp_labels = []

        with open(self.label_path, 'r') as json_file:
            labels_dict = json.load(json_file)

        npy_images_files_path_list = os.listdir(self.file_path)

        for file_name in npy_images_files_path_list:
            if file_name.endswith('.npy'):
                image_path = os.path.join(self.file_path, file_name)

                # Load image. (I get image in pillow)
                image = np.load(image_path)
                image = Image.fromarray(image.astype(np.uint8))

                # Convert it into numpy array
                image = np.array(image).astype(np.float32)

                image = resize(image, image_size)

                temp_images.append(image)
                temp_labels.append(labels_dict[file_name.split('.')[0]])

        self.images = copy.deepcopy(temp_images)
        self.labels = copy.deepcopy(temp_labels)
                # self.labels.append(self.class_dict[labels_dict[file_name.split('.')[0]]])

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        images = []
        labels = []

        start_index = self.current_index
        end_index = self.current_index + self.batch_size
        self.current_index += self.batch_size

        if end_index > len(self.images) and self.current_index > len(self.images):
            # This can also be done but fuck it for now.
            # self.current_index = 0
            self.epoch += 1

        if end_index >= len(self.images) and self.current_index >= len(self.images):
            self.current_index = len(self.images)
            end_index = len(self.images)

        # Note: To uncomment Method 1; you need to Comment out Method 2's two ifs.
        ## Method1 - Different logic : fix the index in case of overflowing
        # if end_index >= len(self.images):
        #     self.current_index = end_index = len(self.images)
        #     start_index = end_index - self.batch_size

        # # Method 2 - Take the remaining data from beginning
        # Opar se "end_index - start_index" utne pakar le.
        if end_index != start_index and start_index < end_index:
            # original_batch_images = np.asarray(self.images[start_index: end_index])
            # original_batch_labels = np.asarray(self.labels[start_index: end_index])
            original_batch_images = self.images[start_index: end_index]
            original_batch_labels = self.labels[start_index: end_index]

        if end_index - start_index < self.batch_size and end_index != start_index:
            # original_batch_images.concatenate(original_batch_images, self.images[0: (self.batch_size - len(original_batch_images))])
            # original_batch_labels.concatenate(original_batch_labels, self.labels[0: (self.batch_size - len(original_batch_labels))])
            original_batch_images.extend(self.images[0: (self.batch_size - len(original_batch_images))])
            original_batch_labels.extend(self.labels[0: (self.batch_size - len(original_batch_labels))])

            # original_batch_images = [item for sublist in original_batch_images for item in sublist]
            # original_batch_labels = [item for sublist in original_batch_labels for item in sublist]


        # Execute only if end_index has reach the end; thus start and end indices are same
        if end_index == start_index and end_index == len(self.images):
            # original_batch_images = np.asarray(self.images[0: end_index])
            # original_batch_labels = np.asarray(self.labels[0: end_index])
            original_batch_images = self.images[0: end_index]
            original_batch_labels = self.labels[0: end_index]

        # Shuffling
        if self.shuffle:
            indices = np.arange(len(original_batch_images))
            np.random.shuffle(indices)
            # Use [0] cuz random().shuffle() returning a tuple
            images = [original_batch_images[i] for i in indices]
            labels = [original_batch_labels[i] for i in indices]

        # Mirroring
        if self.mirroring:
            # # Method 1 : Simple way
            # for i in range(len(original_images)):
            #     if random.random() > 0.5:
            #         images[i] = np.fliplr(original_images[i])
            #         labels[i] = original_labels[i]

            # # Method 2 : Lambda way
            # images = [np.fliplr(img) if random.random() > 0.5 else img for img in original_batch_images]
            # images, labels = (lambda i, j: (np.fliplr(i), np.fliplr(j)) if random.random() > 0.5 else (i, j))(original_batch_images, original_batch_labels)

            # images = np.asarray((lambda i: (np.fliplr(i)) if random.random() > 0.5 else i)(original_batch_images))

            # v1.0 way (wrong)
            images = np.asarray((lambda i: (np.fliplr(i)) if random.random() > 0.5 else i)(original_batch_images))

            # v2.0
            images = np.asarray(list(map(lambda x:  (np.fliplr(x)) if random.random() > 0.5 else x, original_batch_images)))

            # labels = np.asarray(original_batch_labels)
            # images = list(map(lambda i: np.fliplr(i) if random.random() > 0.5 else i, original_images))

        # Rotation
        if self.rotation:
            # # Method 1 : Simple way
            # angles = [1, 2, 3]
            # for i in range(len(images)):
            #     angle = random.choice(angles)
            #     images[i] = np.rot90(images[i], k=angle, axes=(0, 1))

            # # Method 2 : Lamda way
            # k : Number of times the array is rotated by 90 degrees. (copied from official documentation)
            times = [1, 2, 3]
            # single_time_choice = random.choice(times)

            # images = [lambda x: np.rot90(x, k=single_time_choice, axes=(1, 2)), np.asarray(original_batch_images)][1]
            # rotates each image by 90, 180 or 270 degrees along the first two axes, which are (0,1),
            # i.e., the vertical and horizontal axes. This means the image is rotated as if it were a 2D image,
            # without rotating the color channels.
            # images = np.asarray([lambda x: np.rot90(x, k=single_time_choice, axes=(0, 1)), np.asarray(original_batch_images)][1])

            # rotates each image by 90, 180 or 270 degrees along the last two axes, which are (1,2),
            # i.e., the horizontal and depth axes. This means the image is rotated as if it were a 2D
            # image in each color channel separately.
            # images = np.asarray([lambda x: np.rot90(x, k=random.choice(times), axes=(1, 2, 3)), np.asarray(original_batch_images)][1])

            # images = np.asarray([lambda x: np.rot90(x, k=random.choice(times), axes=(1, 2)), np.asarray(original_batch_images)][1])


            # # Method 1: Simple way (correct-working)
            # for i in range(0, len(original_batch_images)):
            #     images.append(np.rot90(original_batch_images[i], k=random.choice(times), axes=(0, 1)))
            #
            # images = np.asarray(images)
            # torF = np.array_equal(images, np.asarray(original_batch_images))

            # Method 2 : lamda way
            images = np.asarray(
                list(map(lambda x: np.rot90(x, k=random.choice(times), axes=(0, 1)), original_batch_images)))

            # labels = np.asarray(original_batch_labels)



        # return images, labels if images == [] and labels == [] else original_batch_images, original_batch_labels
        if (images != [] and labels != []):
            return np.asarray(images), np.asarray(labels)
        elif isinstance(images, np.ndarray) and images.size > 0:
            return np.asarray(images), np.asarray(labels)
        else:
            return np.asarray(original_batch_images), np.asarray(original_batch_labels)

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        # Mirror the image
        if random.random() > 0.5:
            img = np.fliplr(img)

        times = [1, 2, 3]
        # single_time_choice = random.choice(times)

        # Rotate the image
        # k : Number of times the array is rotated by 90 degrees. (copied from official documentation)
        img = np.rot90(img, k=random.choice(times), axes=(0, 1))

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, int_label):
        # This function returns the class name for a specific input
        return self.labels_dict[str(int_label)]
        # return self.class_dict[self.labels_dict[str(int_label)]]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.

        number_of_rows = 4
        number_of_cols = 3

        images, labels = next()

        fig, plt_axs = plt.subplots(nrows=number_of_rows, ncols=number_of_cols, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)

        plt_axs = plt_axs.flatten()

        for i in range(0, number_of_rows):
            for j in range(0, number_of_cols):
                plt_axs[i, j].plot(images[i])
                plt_axs[i, j].set_title(self.class_name(i))
                plt_axs[i, j].axis('off')
                plt_axs[i, j].imshow(images[i])
        plt.show()
