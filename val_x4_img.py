import torch
import cv2
import numpy as np
tseed = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from pytorch_msssim import ssim, ms_ssim

psnr = PeakSignalNoiseRatio()
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
# ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

# read imgx4
imgx4 = cv2.imread('/root/mms/DRCT/results/DRCT64/chicago4_image_onnx_infer.png')
# 转换为RGB
imgx4 = cv2.cvtColor(imgx4, cv2.COLOR_BGR2RGB)
# 转换为浮点型，并归一化到[0, 1]
imgx4 = imgx4.astype(np.float32) / 255.0
imagex4_tensor = torch.from_numpy(imgx4).permute(2, 0, 1).unsqueeze(0)

# readHR
img_HR = cv2.imread('/root/mms/Real-ESRGAN/datasets/L_Aerial_chicago_multiscale/1/chicago4_image.png')
# 转换为RGB
img_HR = cv2.cvtColor(img_HR, cv2.COLOR_BGR2RGB)
# 转换为浮点型，并归一化到[0, 1]
img_HR = img_HR.astype(np.float32) / 255.0
imageHR_tensor = torch.from_numpy(img_HR).permute(2, 0, 1).unsqueeze(0)

print("PSNR:", psnr(imagex4_tensor, imageHR_tensor))
print("LPIPS:", lpips(imagex4_tensor, imageHR_tensor)) # SSIM的取值范围为[0,1]，约接近0表示效果越好，先行sota模型指标通常在0.2左左右

ms_ssim_score = ms_ssim(imagex4_tensor, imageHR_tensor)

# 打印MS-SSIM分数和全局对比度损失
print(f"MS-SSIM: {ms_ssim_score}")

# chicago4_image_onnx_infer.png
# PSNR: tensor(29.6716)
# LPIPS: tensor(0.1132)
# MS-SSIM: 0.9713466167449951