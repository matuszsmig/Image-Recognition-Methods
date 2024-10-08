import torch
import imageio.v3 as iio
import cv2

IMAGE_PATH = './image.jpg'

def read_image():
    return cv2.imread(IMAGE_PATH)

def write_image(image):
    numpy_image = torch.Tensor.numpy(image)
    cv2.imshow(IMAGE_PATH, numpy_image) 

#STEP 1
def devide_image(image):
    torch_tensor = torch.tensor(image)
    return torch.div(torch_tensor, 255)

def add_padding(image, padding):
    return torch.nn.functional.pad(image, padding)

def main():
    img = read_image()
    print(img)
    # devided_image = devide_image(img)
    # padding = (1, 1, 1, 1)
    # image_with_padding = add_padding(devided_image, padding)
    # write_image(image_with_padding)

    # cv2.waitKey(0)        
    # cv2.destroyAllWindows() 

    return 0

if __name__ == "__main__":
    main()