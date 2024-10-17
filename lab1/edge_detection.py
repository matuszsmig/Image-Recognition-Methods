import torch
from torch import nn
import cv2
import numpy as np
import torchvision

IMAGE_PATH = './image2.jpg'
SOBEL_FILTER_HORIZONTAL = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
SOBEL_FILTER_VERTICAL = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

def read_image():
    return cv2.imread(IMAGE_PATH)

def write_image(image):
    image = image.squeeze(0)
    image = image.squeeze(0)
    numpy_image = image.detach().numpy()
    numpy_image = 255 * numpy_image
    numpy_image = numpy_image.astype(np.uint8)
    cv2.imshow(IMAGE_PATH, numpy_image) 

#STEP 1
def devide_image(image):
    torch_tensor = torch.tensor(image)
    return torch.div(torch_tensor, 255)

#STEP 2
def get_gray_image(image):
    conv = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
    weights = torch.tensor([[[[0.3]], [[0.59]], [[0.11]]]])
    conv.weight.data = weights
    image = torch.permute(image, (2, 0, 1)) # zamiana w hwc na nchw
    image = image.unsqueeze(0)
    return conv(image)

#STEP 3
# Zdecydowałem się użyć average pooling, ponieważ że wydaje mi się, że będzie on bardziej adekwatny.
# Mając przykład macierzy 4x4 w której wszytkie wartości mają wartości bliskie 0, poza jedną, która jest bliska 1, to mniejszą stratą,
# będzie wylicznie średniej z nich, niż ustawienie tego piksela na sztywno jako ta wysoka wartość (bliska 1).
def apply_pooling(image):
    pooling = nn.AvgPool2d(4, 4)
    image = pooling(image)
    return image

#STEP 4
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def convolution_with_filter(image, size):
    conv = nn.Conv2d(1, 1, kernel_size=size, stride=1, padding=size//2, bias=False)
    gaussian_filter = torch.from_numpy(gaussian_kernel(size)).float().unsqueeze(0).unsqueeze(0)
    conv.weight.data = gaussian_filter
    return conv(image)

#STEP 5
def apply_sobel_filter(image, filter):
    conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv_weight = filter.float().unsqueeze(0).unsqueeze(0)
    conv.weight.data = conv_weight
    return conv(image)

def merge_images(vertical_image, horizontal_image):
    vertical_image_squared = torch.square(vertical_image)
    horizontal_image_squared = torch.square(horizontal_image)
    summed_images = torch.add(vertical_image_squared, horizontal_image_squared)
    return torch.sqrt(summed_images)

#STEP 6
def non_max_suppression(image, direction):
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.float32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                # angle 0° (horizontal)
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = image[i, j+1]
                    r = image[i, j-1]
                # angle 45° (diagonal up-right)
                elif (22.5 <= angle[i, j] < 67.5):
                    q = image[i+1, j-1]
                    r = image[i-1, j+1]
                # angle 90° (vertical)
                elif (67.5 <= angle[i, j] < 112.5):
                    q = image[i+1, j]
                    r = image[i-1, j]
                # angle 135° (diagonal up-left)
                elif (112.5 <= angle[i, j] < 157.5):
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]

                if image[i, j] >= q and image[i, j] >= r:
                    result[i, j] = image[i, j]
                else:
                    result[i, j] = 0

            except IndexError as e:
                pass

    return result

#STEP 7
def filter_edges(image, threshold=0.0):
    relu = nn.ReLU()
    result = relu(image - threshold)
    result = (result > 0).float()
    
    return result

#STEP 9
def merge_results(cannys_image, base_image):
    output = nn.functional.interpolate(cannys_image, scale_factor=4, mode='nearest') * 255
    output = output.repeat(1, 3, 1, 1)
    base_image = torch.from_numpy(base_image).float().unsqueeze(0).permute(0, 3, 1, 2)
    base_image[:, 1, :, :] = base_image[:, 1, :, :] + output[:, 1, :, :]

    base_image[output > 0] = 0

    return base_image

def write_final_image(image):
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    numpy_image = (image * 255).astype(np.uint8)
    cv2.imshow("Image", numpy_image)

def main():
    img = read_image()
    devided_image = devide_image(img)
    grayscale_image = get_gray_image(devided_image)
    image_after_pooling = apply_pooling(grayscale_image)

    gaussian_blur = convolution_with_filter(image_after_pooling, 3) # Dla image size=5, dla image2 size=3
    horizontal_image_gradient = apply_sobel_filter(gaussian_blur, SOBEL_FILTER_HORIZONTAL)
    vertical_image_gradient = apply_sobel_filter(gaussian_blur, SOBEL_FILTER_VERTICAL)

    merged_images = merge_images(vertical_image_gradient, horizontal_image_gradient)
    #STEP 5.3
    gradient_direction = torch.arctan(vertical_image_gradient / horizontal_image_gradient)

    merged_images_2d = merged_images.squeeze(0).squeeze(0)
    gradient_direction_2d = gradient_direction.squeeze(0).squeeze(0)
    nms_result = non_max_suppression(merged_images_2d, gradient_direction_2d)
    image_after_non_max_suppression = torch.from_numpy(nms_result).float().unsqueeze(0).unsqueeze(0)

    filtred_edges = filter_edges(image_after_non_max_suppression, 0.1)

    final_result = merge_results(filtred_edges, img)

    write_final_image(final_result)
    write_image(filtred_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()