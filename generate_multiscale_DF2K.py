import argparse
import glob
import os
from PIL import Image


def main(args):
    # For DF2K, we consider the following three scales,
    # and the smallest image whose shortest edge is 400
    scale_list = [0.5]
    # scale_list = [0.75, 0.5, 1 / 3]
    shortest_edge = 400

    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        for idx, scale in enumerate(scale_list):
            print(f'\t{scale:.2f}')
            rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            if not os.path.exists(os.path.join(args.output, f'{scale_list[idx]}')):
                # 如果文件夹不存在，使用os.makedirs()创建文件夹
                os.makedirs(os.path.join(args.output, f'{scale_list[idx]}'))
            rlt.save(os.path.join(args.output, f'{scale_list[idx]}', f'{basename}.png'))#T{idx}

        # save the smallest image which the shortest edge is 400
        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)
        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        if not os.path.exists(os.path.join(args.output, 'shortest400')):
            os.makedirs(os.path.join(args.output, 'shortest400'))
        rlt.save(os.path.join(args.output, 'shortest400', f'{basename}.png'))#T{idx+1}


if __name__ == '__main__':
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for DF2K dataset (DIV2K + Flickr 2K)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/root/build_datasets/L_Aerial_chicago/img_dir/val/", help='Input folder')
    parser.add_argument('--output', type=str, default='/root/mms/Real-ESRGAN/datasets/L_Aerial_chicago_multiscale_0.5', help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
