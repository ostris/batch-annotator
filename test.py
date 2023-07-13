import os
import torch
import cv2
import argparse
from tqdm import tqdm
from annotators import annotate, annotators, cleanup_annotators, add_annotators_to_arg_parser


def main():
    parser = argparse.ArgumentParser(
        description="Converts an image to all annotations"
    )
    parser.add_argument("input_img", help="input image")
    parser.add_argument("--res", type=int, default=512,
                        help="Resolution to process at. -1 for original size (be careful for large images!)")
    add_annotators_to_arg_parser(parser)
    args = parser.parse_args()

    img = cv2.imread(args.input_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for annotator in tqdm(annotators):
        with torch.no_grad():
            args.annotator = annotator.slug
            # append annotator to output filename
            file_path_without_ext, ext = os.path.splitext(args.input_img)
            out_path = file_path_without_ext + "_" + annotator.slug + ext
            output = annotate(img, args)

            output = output.astype('uint8')
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            cv2.imwrite(out_path, output)
            annotator.cleanup()


    print("FIN")


if __name__ == "__main__":
    main()
