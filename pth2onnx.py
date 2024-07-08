import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet
from drct.archs.DRCT_arch import *

def main(args):
    # An instance of the model
    # set up model (DRCT-L)
    # model = DRCT(upscale=4, in_chans=3,  img_size= 64, window_size= 16, compress_ratio= 3,squeeze_factor= 30,
    #                     conv_scale= 0.01, overlap_ratio= 0.5, img_range= 1., depths= [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    #                     embed_dim= 180, num_heads= [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], gc= 32,
    #                     mlp_ratio= 2, upsampler= 'pixelshuffle', resi_connection= '1conv')

    # set up model (DRCT)
    model = DRCT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio= 2,
        upsampler= 'pixelshuffle',
        resi_connection= '1conv')

    if args.params:
        keyname = 'params'
    else:
        keyname = 'params_ema'
    model.load_state_dict(torch.load(args.input)[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    # An example input
    x = torch.rand(1, 3, args.input_size, args.input_size)
    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, args.output, opset_version=11, export_params=True)
    print(torch_out.shape)


if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='/root/mms/DRCT/experiments/train_DRCT_SRx4_from_scratch/models/net_g_10000.pth', help='Input model path')
    parser.add_argument('--input_size', type=int, default=64, help='network input img size, default 64')
    parser.add_argument('--output', type=str, default='DRCT_X4_inputsize64.onnx', help='Output onnx path')
    parser.add_argument('--params', action='store_false', default=False, help='Use params instead of params_ema')
    args = parser.parse_args()

    main(args)
    print("Done!")
