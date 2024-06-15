import torch
import numpy as np
import copy
from skimage.segmentation import slic, quickshift
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from torchvision import transforms
from PIL import Image
import skimage.io
import matplotlib.pyplot as plt


def view_result(image, coeff, num_superpixels, superpixels):
    num_top_features = 5
    top_features = np.argsort(coeff)[-num_top_features:]

    mask = np.zeros(num_superpixels)
    mask[top_features] = True  # Activate top superpixels
    skimage.io.imshow(perturb_image_v2(image / 2 + 0.5, mask, superpixels))


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

    perturbed_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    superpixels_or_segments = quickshift(image, kernel_size=3, ratio=0.4)

    num_superpixels = np.unique(superpixels_or_segments).shape[0]

    # Generate perturbations
    perturbations = np.random.binomial(1, 0.5, size=(perturbation_times, num_superpixels))

    perturbed_images = []
    for i in range(perturbation_times):
        perturbed_image = perturb_image_v2(np.array(image), perturbations[i], superpixels_or_segments)
        # skimage.io.imshow(perturbed_image/2+0.5)
        perturbed_image_tensor = perturbed_preprocess(perturbed_image)
        perturbed_images.append(perturbed_image_tensor.unsqueeze(0))

    perturbed_images_tensor = torch.cat(perturbed_images)

    input_preprocess_image = preprocess(image)
    input_tensor = input_preprocess_image.unsqueeze(0)

    batch = torch.from_numpy(np.concatenate((input_tensor, perturbed_images_tensor), axis=0))

    with torch.no_grad():
        predictions = model(batch) # .detach().numpy()

    # top_pred_classes = np.argsort(predictions[0])[-5:][::1] # Save ids of top 5 classes

    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    # print(predictions[0])

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    # print(probabilities)

    return predictions, perturbations, num_superpixels, superpixels_or_segments, perturbed_images


def lime_explanation(image, model, perturbation_times=10):
    (predictions, perturbations, num_superpixels, superpixels_or_segments, perturbed_images) = preprocess_and_predict(image, model, perturbation_times)

    original_image = np.ones(num_superpixels)[np.newaxis, :]

    distances = pairwise_distances(perturbations, original_image, metric='euclidean').ravel() # .reshape(1,-1)

    weights = np.exp(-distances / np.std(distances)) #.reshape(1,-1)
    # weights = np.transpose(np.exp(-distances / np.std(distances)) #.reshape(1,-1)

    top_pred_classes = predictions[0].argsort()[-5:][::1]
    class_to_explain = top_pred_classes[0]

    interpretable_model = LinearRegression()
    # interpretable_model.fit(perturbations, np.squeeze(predictions, axis=1), sample_weight=weights)
    # interpretable_model.fit(perturbations, predictions[1:].detach().numpy(), sample_weight=weights)
    interpretable_model.fit(X=perturbations, y=predictions[1:, class_to_explain], sample_weight=weights)

    explanation = interpretable_model.coef_

    # num_top_features = 4
    # view_result(image, explanation, num_superpixels, superpixels_or_segments)

    return explanation, perturbed_images


if __name__ == "__main__":
    # Load image data
    img1 = Image.open('.\\images\\1024px-Schloss-Erlangen02.jpg')
    img2 = Image.open('.\\images\\1024px-Alte-universitaets-bibliothek_universitaet-erlangen.jpg')
    img3 = Image.open('.\\images\\1024px-Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.jpg')

    img1.show()

    # img1 = plt.imread('.\\images\\1024px-Schloss-Erlangen02.jpg')
    # img2 = plt.imread('.\\images\\1024px-Alte-universitaets-bibliothek_universitaet-erlangen.jpg')
    # img3 = plt.imread('.\\images\\1024px-Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.jpg')

    # Load Inception v3 neural network model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

    model.aux_logits = False

    # model.drop_last = True
    perturbation_time = 10
    explanation, perturbed_images = lime_explanation(img1, model, perturbation_times=perturbation_time)

    plt.figure()
    for i in range (perturbation_time):
        # skimage.io.imshow(perturbed_images[i])
        perturbed_squeezed_tensor = torch.squeeze(perturbed_images[i], dim=0)
        plt.imshow((perturbed_squeezed_tensor * 255).permute(1, 2, 0).byte().numpy()) # .dtype(np.uint8))
        plt.axis('off')  # Turn off axis labels
        plt.show()

    print("Explanation:", explanation)




def get_prediction_labels(probabilities):
    file_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    # Download ImageNet labels

    # Read the categories
    with open("..//imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())