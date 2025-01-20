import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import os
import math
import cv2
from torchvision.utils import make_grid
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

_l1_loss = nn.L1Loss()

def mae(input, target):
    """Calculate Mean Absolute Error.
    
    Args:
        input (torch.Tensor): Input tensor
        target (torch.Tensor): Target tensor
    
    Returns:
        float: MAE value
    """
    with torch.no_grad():
        output = _l1_loss(input, target)
    return output


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(n_dim)
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def load_image(image_path, grayscale=False):
    """Load an image with OpenCV.
    
    Args:
        image_path (str): Path to the image
        grayscale (bool): If True, load as grayscale
    
    Returns:
        numpy.ndarray: Loaded image
    """
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image not found or format not supported: {image_path}")
    return img


def cal_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    if isinstance(img1, torch.Tensor):
        img1 = tensor2img(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2img(img2)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def cal_ssim(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1 = tensor2img(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2img(img2)
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def cal_psnr_folder(folder1, folder2):
    """Calculate PSNR for all images in two folders.
    
    Args:
        folder1 (str): Path to first folder
        folder2 (str): Path to second folder
    
    Returns:
        list: PSNR values for each image pair
    """
    filenames1 = sorted(os.listdir(folder1))
    filenames2 = sorted(os.listdir(folder2))
    
    if len(filenames1) != len(filenames2):
        raise ValueError(f"Folders contain different numbers of images: {len(filenames1)} vs {len(filenames2)}")
    
    psnr_values = []
    for file1, file2 in zip(filenames1, filenames2):
        img1_path = os.path.join(folder1, file1)
        img2_path = os.path.join(folder2, file2)
        try:
            img1 = load_image(img1_path, grayscale=True)
            img2 = load_image(img2_path, grayscale=True)
            psnr_values.append(cal_psnr(img1, img2))
        except Exception as e:
            print(f"Error processing {file1} and {file2}: {str(e)}")
            continue
    return psnr_values


def cal_ssim_folder(folder1, folder2):
    """Calculate SSIM for all images in two folders.
    
    Args:
        folder1 (str): Path to first folder
        folder2 (str): Path to second folder
    
    Returns:
        list: SSIM values for each image pair
    """
    filenames1 = sorted(os.listdir(folder1))
    filenames2 = sorted(os.listdir(folder2))
    
    if len(filenames1) != len(filenames2):
        raise ValueError(f"Folders contain different numbers of images: {len(filenames1)} vs {len(filenames2)}")
    
    ssim_values = []
    for file1, file2 in zip(filenames1, filenames2):
        img1_path = os.path.join(folder1, file1)
        img2_path = os.path.join(folder2, file2)
        try:
            img1 = load_image(img1_path, grayscale=True)
            img2 = load_image(img2_path, grayscale=True)
            ssim_values.append(cal_ssim(img1, img2))
        except Exception as e:
            print(f"Error processing {file1} and {file2}: {str(e)}")
            continue
    return ssim_values
