import os
import sys
import cv2
import torch

CONTROL_NET_ROOT = os.path.join(os.path.dirname(__file__), 'repositories', 'controlnet')
sys.path.append(CONTROL_NET_ROOT)

from annotator.util import resize_image, HWC3

annotator_list = [
    'canny',
    'hed',
    'pidi',
    'mlsd',
    'midas',
    'zoe',
    'normalbae',
    'openpose',
    'uniformer',
    'lineart_anime',
    'lineart',
    'oneformer_coco',
    'oneformer_ade20k',
    'content_shuffler',
    'color_shuffler',
]

model_canny = None


def canny(img, res, l=100, h=200):
    img = resize_image(HWC3(img), res)
    global model_canny
    if model_canny is None:
        from annotator.canny import CannyDetector
        model_canny = CannyDetector()
    result = model_canny(img, l, h)
    return [result]


model_hed = None


def hed(img, res):
    img = resize_image(HWC3(img), res)
    global model_hed
    if model_hed is None:
        from annotator.hed import HEDdetector
        model_hed = HEDdetector()
    result = model_hed(img)
    return [result]


model_pidi = None


def pidi(img, res):
    img = resize_image(HWC3(img), res)
    global model_pidi
    if model_pidi is None:
        from annotator.pidinet import PidiNetDetector
        model_pidi = PidiNetDetector()
    result = model_pidi(img)
    return [result]


model_mlsd = None


def mlsd(img, res, thr_v=0.1, thr_d=0.1):
    img = resize_image(HWC3(img), res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import MLSDdetector
        model_mlsd = MLSDdetector()
    result = model_mlsd(img, thr_v, thr_d)
    return [result]


model_midas = None


def midas(img, res):
    img = resize_image(HWC3(img), res)
    global model_midas
    if model_midas is None:
        from annotator.midas import MidasDetector
        model_midas = MidasDetector()
    result = model_midas(img)
    return [result]


model_zoe = None


def zoe(img, res):
    img = resize_image(HWC3(img), res)
    global model_zoe
    if model_zoe is None:
        from annotator.zoe import ZoeDetector
        model_zoe = ZoeDetector()
    result = model_zoe(img)
    return [result]


model_normalbae = None


def normalbae(img, res):
    img = resize_image(HWC3(img), res)
    global model_normalbae
    if model_normalbae is None:
        from annotator.normalbae import NormalBaeDetector
        model_normalbae = NormalBaeDetector()
    result = model_normalbae(img)
    return [result]


model_openpose = None


def openpose(img, res, hand_and_face=True):
    img = resize_image(HWC3(img), res)
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import OpenposeDetector
        model_openpose = OpenposeDetector()
    result = model_openpose(img, hand_and_face)
    return [result]


model_uniformer = None


def uniformer(img, res):
    img = resize_image(HWC3(img), res)
    global model_uniformer
    if model_uniformer is None:
        from annotator.uniformer import UniformerDetector
        model_uniformer = UniformerDetector()
    result = model_uniformer(img)
    return [result]


model_lineart_anime = None


def lineart_anime(img, res):
    img = resize_image(HWC3(img), res)
    global model_lineart_anime
    if model_lineart_anime is None:
        from annotator.lineart_anime import LineartAnimeDetector
        model_lineart_anime = LineartAnimeDetector()
    result = model_lineart_anime(img)
    return [result]


model_lineart = None


def lineart(img, res, coarse=False):
    img = resize_image(HWC3(img), res)
    global model_lineart
    if model_lineart is None:
        from annotator.lineart import LineartDetector
        model_lineart = LineartDetector()
    result = model_lineart(img, coarse)
    return [result]


model_oneformer_coco = None


def oneformer_coco(img, res):
    img = resize_image(HWC3(img), res)
    global model_oneformer_coco
    if model_oneformer_coco is None:
        from annotator.oneformer import OneformerCOCODetector
        model_oneformer_coco = OneformerCOCODetector()
    result = model_oneformer_coco(img)
    return [result]


model_oneformer_ade20k = None


def oneformer_ade20k(img, res):
    img = resize_image(HWC3(img), res)
    global model_oneformer_ade20k
    if model_oneformer_ade20k is None:
        from annotator.oneformer import OneformerADE20kDetector
        model_oneformer_ade20k = OneformerADE20kDetector()
    result = model_oneformer_ade20k(img)
    return [result]


model_content_shuffler = None


def content_shuffler(img, res):
    img = resize_image(HWC3(img), res)
    global model_content_shuffler
    if model_content_shuffler is None:
        from annotator.shuffle import ContentShuffleDetector
        model_content_shuffler = ContentShuffleDetector()
    result = model_content_shuffler(img)
    return [result]


model_color_shuffler = None


def color_shuffler(img, res):
    img = resize_image(HWC3(img), res)
    global model_color_shuffler
    if model_color_shuffler is None:
        from annotator.shuffle import ColorShuffleDetector
        model_color_shuffler = ColorShuffleDetector()
    result = model_color_shuffler(img)
    return [result]


def post_process(annotated_img, original_image):
    img = annotated_img
    # is is list, get the first one
    if isinstance(img, list):
        img = img[0]
    H, W, _ = original_image.shape
    img = HWC3(img)
    output_img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    # make sure it is rgba
    if output_img.ndim == 2:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)
    return output_img


def cleanup_annotators():
    annotator_model_list = [
        model_hed,
        model_pidi,
        model_mlsd,
        model_midas,
        model_zoe,
        model_normalbae,
        model_openpose,
        model_uniformer,
        model_lineart_anime,
        model_lineart,
        model_oneformer_coco,
        model_oneformer_ade20k,
        model_content_shuffler,
        model_color_shuffler,
    ]
    for model in annotator_model_list:
        if model is not None:
            del model


def annotate(input_image, args):
    an = args.annotator
    res = args.res
    # clone numpy image


    img = input_image.copy()
    with torch.no_grad():
        if an == 'canny':
            low_threshold = 100
            high_threshold = 200
            out = canny(img, res, low_threshold, high_threshold)
        elif an == 'hed':
            out = hed(img, res)
        elif an == 'pidi':
            out = pidi(img, res)
        elif an == 'mlsd':
            out = mlsd(img, res)
        elif an == 'midas':
            out = midas(img, res)
        elif an == 'zoe':
            out = zoe(img, res)
        elif an == 'normalbae':
            out = normalbae(img, res)
        elif an == 'openpose':
            out = openpose(img, res)
        elif an == 'uniformer':
            out = uniformer(img, res)
        elif an == 'lineart_anime':
            out = lineart_anime(img, res)
        elif an == 'lineart':
            out = lineart(img, res)
        elif an == 'oneformer_coco':
            out = oneformer_coco(img, res)
        elif an == 'oneformer_ade20k':
            out = oneformer_ade20k(img, res)
        elif an == 'content_shuffler':
            out = content_shuffler(img, res)
        elif an == 'color_shuffler':
            out = color_shuffler(img, res)
        else:
            raise ValueError(f'Unknown annotator: {an}')

    out = post_process(out, input_image)
    return out
