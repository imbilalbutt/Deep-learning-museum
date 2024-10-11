import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.segmentation import slic, quickshift
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from torchvision import transforms
from PIL import Image

# Load image data
img1 = Image.open('.\\images\\1024px-Schloss-Erlangen02.jpg')
img2 = Image.open('.\\images\\1024px-Alte-universitaets-bibliothek_universitaet-erlangen.jpg')
img3 = Image.open('.\\images\\1024px-Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.jpg')

# Load Inception v3 neural network model
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)


# def perturb_image_v1(image, superpixels_or_segments):
#     # Generate segmentation for image
#     fudged_image = image.copy()
#
#     # num_superpixels = np.unique(superpixels_or_segments).shape[0]
#
#     for x in np.unique(superpixels_or_segments):
#         fudged_image[superpixels_or_segments == x] = (
#             np.mean(image[superpixels_or_segments == x][:, 0]),
#             np.mean(image[superpixels_or_segments == x][:, 1]),
#             np.mean(image[superpixels_or_segments == x][:, 2]))
#
#     plt.imshow(image)
#
#     plt.imshow(fudged_image)
#
#     return fudged_image, superpixels_or_segments


def perturb_image_v2(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]

    return perturbed_image


def predict(image, model, perturbation_times):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    original_image_tensor = preprocess(image)
    # input_batch = original_image_tensor.unsqueeze(0)
    # np.random.seed(222)
    # original_prediction = model(input_batch)

    # top_pred_classes = original_prediction[0].argsort()[-5:][::-1]     #Index of top 5 classes
    # decode_predictions(original_prediction)[0]  # Print top 5 classes

    superpixels_or_segments = quickshift(image, kernel_size=3, max_dist=200, ratio=0.4)

    num_superpixels = np.unique(superpixels_or_segments).shape[0]
    top_pred_classes = 0

    # Generate perturbations
    perturbations = np.random.binomial(1, 0.5, size=(perturbation_times, num_superpixels))

    perturbed_images = []
    predictions = []

    pertubed_preprocess = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        # transforms.ToTensor()
    ])

    for i in range(perturbation_times):
        perturbed_image = perturb_image_v2(image, perturbations[i], superpixels_or_segments)
        perturbed_images.append(perturbed_image)

    perturbed_images_tensor = np.stack(perturbed_images)
    # input_batch = perturbed_images_tensor.unsqueeze(0)

    perturbed_image_tensor = pertubed_preprocess(perturbed_images_tensor)
    np.random.seed(222)
    prediction = model(perturbed_image_tensor)

    top_pred_classes = prediction[0].argsort()[-5:][::1]  # Save ids of top 5 classes

    predictions.append(prediction.detach().numpy())

    # perturbed_images = np.array(perturbed_images) # Convert from (701, 1024, 3) ---> to (1, 701, 1024, 3)
    # predictions = np.array(predictions)  # .detach().numpy())

    return predictions, top_pred_classes, perturbations, num_superpixels


def lime_explanation(image, model, perturbation_times=10):
    # perturbed_images,
    (predictions, top_pred_classes, perturbations, num_superpixels) = predict(image, model, perturbation_times)
    # original_image_flat = image.flatten().reshape(1, -1)

    # perturbed_images_flat - original_image_flat EVALUATE
    # perturbed_images_flat = []
    # for i in range(perturbation_times):
    #     perturbed_images_flat.append(perturbed_images[i].reshape(1, -1))

    # distances = np.empty((1, perturbation_times), dtype=float, order='C', like=None) # np.zeros((1,
    # perturbation_times)) for i in range(perturbation_times): distance = pairwise_distances(perturbed_images_flat[
    # i], original_image_flat, metric='euclidean').ravel()
    original_image = np.ones(num_superpixels)[np.newaxis, :]  #Perturbation with all superpixels enabled
    distances = pairwise_distances(perturbations, original_image, metric='euclidean').ravel()
    # if i is 0:
    #     distances = distance
    # else:
    #     distances = np.append(distances, distance, axis=0)
    weights = np.exp(-distances / np.std(distances))

    # kernel_width = 0.25
    # weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function

    interpretable_model = LinearRegression()
    interpretable_model.fit(perturbations, np.squeeze(predictions, axis=1), sample_weight=weights)

    # .fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
    # coeff = interpretable_model.coef_[0]

    explanation = interpretable_model.coef_

    return explanation.reshape(image.shape)


if __name__ == "__main__":
    # Xi = resize(img1, (299, 299))
    # Xi = (Xi - 0.5) * 2  # Inception pre-processing

    # image = np.array(img1)
    model.aux_logits = False
    model.drop_last = True
    explanation = lime_explanation(img1, model, perturbation_times=10)

    print("Explanation:", explanation)
