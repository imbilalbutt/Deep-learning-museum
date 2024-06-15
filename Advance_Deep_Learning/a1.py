import torch
import numpy as np
import copy
from skimage.segmentation import slic, quickshift
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from torchvision import transforms
from PIL import Image
# import matplotlib.pyplot as plt


def perturb_image_v2(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]

    return perturbed_image


def preprocess_and_predict(image, model, perturbation_times):

    preprocess = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preturbed_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_batch = preprocess(image)
    input_tensor = input_batch.unsqueeze(0)

    # original_image_tensor_copy = copy.deepcopy(image)
    # batches = np.stack((original_image_tensor, original_image_tensor_copy), axis=0)
    # batches_tensor = torch.from_numpy(batches).float()

    # image_copy = copy.deepcopy(image)
    # batches = np.stack((image_copy, image))
    # batches_tensor = preprocess(batches)

    # np.random.seed(222)
    # original_prediction = model(batches_tensor)
    # # original_prediction = model(input_batch)

    # with torch.no_grad():
    #     np.random.seed(222)
    #     output = model(input_batch)

    superpixels_or_segments = quickshift(image, kernel_size=3, ratio=0.4)

    num_superpixels = np.unique(superpixels_or_segments).shape[0]

    # Generate perturbations
    perturbations = np.random.binomial(1, 0.5, size=(perturbation_times, num_superpixels))

    perturbed_images = []
    predictions = []



    # for i in range(perturbation_times):
    #     perturbed_image = perturb_image_v2(image, perturbations[i], superpixels_or_segments)
    #     perturbed_images.append(perturbed_image)
    #
    # perturbed_images_tensor = np.stack(perturbed_images)
    # # input_batch = perturbed_images_tensor.unsqueeze(0)
    #
    # perturbed_image_tensor = pertubed_preprocess(perturbed_images_tensor)
    # np.random.seed(222)
    # prediction = model(perturbed_image_tensor)

    for i in range(perturbation_times):
        perturbed_image = perturb_image_v2(np.array(image), perturbations[i], superpixels_or_segments)
        perturbed_image_tensor = preturbed_preprocess(perturbed_image)
        perturbed_images.append(perturbed_image_tensor.unsqueeze(0))

    perturbed_images_tensor = torch.cat(perturbed_images)
    batch = torch.from_numpy(np.concatenate((input_tensor, perturbed_images_tensor), axis=0))

    with torch.no_grad():
        predictions = model(batch) # .detach().numpy()

    # top_pred_classes = np.argsort(predictions[0])[-5:][::-1] # Save ids of top 5 classes

    # predictions.append(prediction.detach().numpy())

    return predictions, perturbations, num_superpixels


def lime_explanation(image, model, perturbation_times=10):
    (predictions, perturbations, num_superpixels) = preprocess_and_predict(image, model, perturbation_times)

    original_image = np.ones(num_superpixels)[np.newaxis, :]

    distances = pairwise_distances(perturbations, original_image, metric='euclidean').ravel().reshape(1,-1)

    weights = np.exp(-distances / np.std(distances)).reshape(1,-1)

    interpretable_model = LinearRegression()
    interpretable_model.fit(perturbations, np.squeeze(predictions, axis=1), sample_weight=weights)

    explanation = interpretable_model.coef_

    return explanation.reshape(image.shape)


if __name__ == "__main__":
    # Load image data
    img1 = Image.open('.\\images\\1024px-Schloss-Erlangen02.jpg')
    img2 = Image.open('.\\images\\1024px-Alte-universitaets-bibliothek_universitaet-erlangen.jpg')
    img3 = Image.open('.\\images\\1024px-Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.jpg')

    # img1 = plt.imread('.\\images\\1024px-Schloss-Erlangen02.jpg')
    # img2 = plt.imread('.\\images\\1024px-Alte-universitaets-bibliothek_universitaet-erlangen.jpg')
    # img3 = plt.imread('.\\images\\1024px-Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.jpg')

    # Load Inception v3 neural network model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

    # model.aux_logits = False
    # model.drop_last = True
    explanation = lime_explanation(img1, model, perturbation_times=10)

    print("Explanation:", explanation)
