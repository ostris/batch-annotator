import os
import torch
import cv2
import argparse
from tqdm import tqdm
from annotators import annotate, annotator_list, cleanup_annotators


def main():
    parser = argparse.ArgumentParser(
        description="Batch converts images to depth map using MiDaS"
    )
    parser.add_argument("input_img", help="input image")
    args = parser.parse_args()

    img = cv2.imread(args.input_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    args.res = 512

    for annotator in tqdm(annotator_list):
        with torch.no_grad():
            args.annotator = annotator
            # append annotator to output filename
            file_path_without_ext, ext = os.path.splitext(args.input_img)
            out_path = file_path_without_ext + "_" + annotator + ext
            output = annotate(img, args)

            output = output.astype('uint8')
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            cv2.imwrite(out_path, output)

            cleanup_annotators()
            torch.cuda.empty_cache()

    print("FIN")


if __name__ == "__main__":
    main()
