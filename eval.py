import argparse
import numpy as np
from core.base_dataset import BaseDataset
from models.metric import inception_score, cal_psnr_folder, cal_ssim_folder
from pytorch_fid import fid_score as pytorch_fid
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, help='Generate images directory')
   
    ''' parser configs '''
    args = parser.parse_args()

    fid_score = pytorch_fid.calculate_fid_given_paths(
        [args.src, args.dst],
        batch_size=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048
    )
    is_mean, is_std = inception_score(BaseDataset(args.dst), cuda=True, batch_size=8, resize=True, splits=10)

    psnr = cal_psnr_folder(args.src, args.dst)
    ssim = cal_ssim_folder(args.src, args.dst)
    
    print('FID: {}'.format(fid_score))
    print('IS:{}'.format(is_mean))
    print('IS STD: {}'.format(is_std))
    print('PSNR: {}'.format(np.mean(psnr)))
    print('PSNR STD: {}'.format(np.std(psnr)))
    print('SSIM: {}'.format(np.mean(ssim)))
    print('SSIM STD: {}'.format(np.std(ssim)))
