import os
import sys
import cv2
import numpy as np
import torch
import importlib

CONTROL_NET_ROOT = os.path.join(os.path.dirname(__file__), 'repositories', 'controlnet')
sys.path.append(CONTROL_NET_ROOT)

from annotator.util import resize_image, HWC3

annotators = []


def value_map(x, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def add_annotators_to_arg_parser(parser):
    # add arguments for each annotator
    for annotator in annotators:
        if annotator.additional_args is not None:
            for arg in annotator.additional_args:
                # handle booleans
                if arg["type"] == bool:
                    parser.add_argument(f"--{arg['slug']}", action="store_true", help=arg["help"])
                else:
                    parser.add_argument(
                        f"--{arg['slug']}", type=arg['type'], default=arg['default'] if 'default' in arg else None,
                        help=arg['help'])


class Annotator:
    def __init__(
            self,
            name,
            slug=None,
            import_path=None,
            import_class_name=None,
            additional_args=None,
            call_override=None
    ):
        self.name = name
        self.slug = slug
        if self.slug is None:
            self.slug = self.name
        self.model = None
        self.import_path = import_path
        self.import_class_name = import_class_name
        self.additional_args = additional_args
        if self.import_class_name is None and call_override is None:
            raise ValueError('import_class_name must be specified for Annotator: ' + self.name)
        if self.import_path is None:
            self.import_path = 'annotator.' + self.slug

        self.call_override = call_override

    def __call__(self, img, res, *args, **kwargs):
        if self.call_override is not None:
            return self.call_override(self, img, res, *args, **kwargs)
        if self.model is None:
            self.load()
        img = resize_image(HWC3(img), res)
        res = self.model(img, *args, **kwargs)
        return [res]

    def load(self):
        module = importlib.import_module(self.import_path)
        annotator_model = getattr(module, self.import_class_name)
        self.model = annotator_model()

    def cleanup(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None


annotators.append(
    Annotator(
        name='Canny',
        slug='canny',
        import_path='annotator.canny',
        import_class_name='CannyDetector',
        additional_args=[
            {
                'slug': 'canny_low_threshold',
                'keyword': 'low_threshold',
                'type': int,
                'default': 100,
                'help': 'Low threshold for Canny edge detection'
            },
            {
                'slug': 'canny_high_threshold',
                'keyword': 'high_threshold',
                'type': int,
                'default': 200,
                'help': 'High threshold for Canny edge detection'
            }
        ]
    )
)

annotators.append(
    Annotator(
        name='HED',
        slug='hed',
        import_path='annotator.hed',
        import_class_name='HEDdetector'
    )
)

annotators.append(
    Annotator(
        name='PIDI',
        slug='pidi',
        import_path='annotator.pidinet',
        import_class_name='PidiNetDetector'
    )
)

annotators.append(
    Annotator(
        name='MLSD Line Detection',
        slug='mlsd',
        import_path='annotator.mlsd',
        import_class_name='MLSDdetector',
        additional_args=[
            {
                'slug': 'mlsd_score_thr',
                'keyword': 'thr_v',
                'type': float,
                'default': 0.1,
                'help': 'Threshold for score of line detection'
            },
            {
                'slug': 'mlsd_dist_thr',
                'keyword': 'thr_d',
                'type': float,
                'default': 0.1,
                'help': 'Threshold for distance of line detection'
            }
        ]
    )
)

annotators.append(
    Annotator(
        name='Midas Depth',
        slug='midas',
        import_path='annotator.midas',
        import_class_name='MidasDetector'
    )
)

annotators.append(
    Annotator(
        name='Zoe Depth',
        slug='zoe',
        import_path='annotator.zoe',
        import_class_name='ZoeDetector'
    )
)

annotators.append(
    Annotator(
        name='NormalBae',
        slug='normalbae',
        import_path='annotator.normalbae',
        import_class_name='NormalBaeDetector'
    )
)

annotators.append(
    Annotator(
        name='OpenPose',
        slug='openpose',
        import_path='annotator.openpose',
        import_class_name='OpenposeDetector',
        additional_args=[
            {
                'slug': 'openpose_hand_and_face',
                'keyword': 'hand_and_face',
                'type': bool,
                'help': 'Whether to detect hand and face'
            }
        ]
    )
)

annotators.append(
    Annotator(
        name='Uniformer',
        slug='uniformer',
        import_path='annotator.uniformer',
        import_class_name='UniformerDetector',
    )
)

annotators.append(
    Annotator(
        name='Lineart Anime',
        slug='lineart_anime',
        import_path='annotator.lineart_anime',
        import_class_name='LineartAnimeDetector'
    )
)

annotators.append(
    Annotator(
        name='Lineart',
        slug='lineart',
        import_path='annotator.lineart',
        import_class_name='LineartDetector',
        additional_args=[
            {
                'slug': 'lineart_coarse',
                'keyword': 'coarse',
                'type': bool,
                'help': 'Whether to use coarse model'
            }
        ]
    )
)

annotators.append(
    Annotator(
        name='Oneformer COCO',
        slug='oneformer_coco',
        import_path='annotator.oneformer',
        import_class_name='OneformerCOCODetector'
    )
)

annotators.append(
    Annotator(
        name='Oneformer ADE20k',
        slug='oneformer_ade20k',
        import_path='annotator.oneformer',
        import_class_name='OneformerADE20kDetector'
    )
)

annotators.append(
    Annotator(
        name='Content Shuffler',
        slug='content_shuffler',
        import_path='annotator.shuffle',
        import_class_name='ContentShuffleDetector'
    )
)

annotators.append(
    Annotator(
        name='Color Shuffler',
        slug='color_shuffler',
        import_path='annotator.shuffle',
        import_class_name='ColorShuffleDetector'
    )
)


# Midas min sets the darkest overlay value. Since it scales from 0.0 to 1.0, 0.1 to 0.5 is a good min value
# To keep the farthest objects visible, we need to adjust the midas values to be higher
def midas_ade20k(self, img, res, midas_ade20k_min=0.5):
    # find midas and ade20k

    midas = None
    oneformer_ade20k = None
    for annotator in annotators:
        if annotator.slug == 'midas':
            midas = annotator
        if annotator.slug == 'oneformer_ade20k':
            oneformer_ade20k = annotator

    midas_imd = midas(img, res)[0]
    ade20k_img = oneformer_ade20k(img, res)[0]

    # expand to 3 channels
    if midas_imd.ndim == 2:
        midas_imd = np.expand_dims(midas_imd, axis=-1)
        # stack
        midas_imd = np.concatenate([midas_imd, midas_imd, midas_imd], axis=-1)

    # convert to 0 - 1 float
    midas_img = midas_imd.astype(np.float32) / 255.0
    ade20k_img = ade20k_img.astype(np.float32) / 255.0

    # adjust midas min value
    midas_img = value_map(midas_img, 0, 1.0, midas_ade20k_min, 1.0)

    merged = ade20k_img * midas_img
    merged = np.clip(merged, 0, 1) * 255
    merged = merged.astype(np.uint8)

    return [merged]


annotators.append(
    Annotator(
        name='Midas + Oneformer ADE20k',
        slug='midas_ade20k',
        additional_args=[
            {
                'slug': 'midas_ade20k_min',
                'type': float,
                'default': 0.2,
                'help': 'Minimum value for midas overlay'
            }
        ],
        call_override=midas_ade20k
    )
)


def normalbae_ade20k(self, img, res, normalbae_ade20k_min=0.5):
    # find midas and ade20k

    oneformer_ade20k = None
    normalbae = None
    for annotator in annotators:
        if annotator.slug == 'normalbae':
            normalbae = annotator
        if annotator.slug == 'oneformer_ade20k':
            oneformer_ade20k = annotator

    normalbae_img = normalbae(img, res)[0]
    ade20k_img = oneformer_ade20k(img, res)[0]


    # convert to 0 - 1 float
    normalbae_img = normalbae_img.astype(np.float32) / 255.0
    ade20k_img = ade20k_img.astype(np.float32) / 255.0

    # make it grayscale by averaging the channels
    if normalbae_img.ndim == 3:
        normalbae_img = np.mean(normalbae_img, axis=-1, keepdims=True)
        # stack
        normalbae_img = np.concatenate([normalbae_img, normalbae_img, normalbae_img], axis=-1)

        # normalize
        normalbae_img = value_map(normalbae_img, np.min(normalbae_img), np.max(normalbae_img), 0, 1)

    # adjust midas min value
    normalbae_img = value_map(normalbae_img, 0, 1.0, normalbae_ade20k_min, 1.0)

    merged = ade20k_img * normalbae_img
    merged = np.clip(merged, 0, 1) * 255
    merged = merged.astype(np.uint8)

    return [merged]


annotators.append(
    Annotator(
        name='Normal Bae + Oneformer ADE20k',
        slug='normalbae_ade20k',
        additional_args=[
            {
                'slug': 'normalbae_ade20k_min',
                'type': float,
                'default': 0.2,
                'help': 'Minimum value for normal bae overlay'
            }
        ],
        call_override=normalbae_ade20k
    )
)


def kitchen_sink(self, img, res):

    min_midas = 0.2
    min_normalbae = 0.5
    min_depth_scale = 0.2
    # find midas and ade20k

    midas = None
    normalbae = None
    oneformer_ade20k = None
    openpose = None
    for annotator in annotators:
        if annotator.slug == 'normalbae':
            normalbae = annotator
        if annotator.slug == 'oneformer_ade20k':
            oneformer_ade20k = annotator
        if annotator.slug == 'midas':
            midas = annotator
        if annotator.slug == 'openpose':
            openpose = annotator


    normalbae_img = normalbae(img, res)[0]
    ade20k_img = oneformer_ade20k(img, res)[0]
    midas_img = midas(img, res)[0]
    openpose_img = openpose(img, res)[0]


    # convert to 0 - 1 float
    normalbae_img = normalbae_img.astype(np.float32) / 255.0
    ade20k_img = ade20k_img.astype(np.float32) / 255.0

    # make it grayscale by averaging the channels
    if normalbae_img.ndim == 3:
        normalbae_img = np.mean(normalbae_img, axis=-1, keepdims=True)
        # stack
        normalbae_img = np.concatenate([normalbae_img, normalbae_img, normalbae_img], axis=-1)

    # expand to 3 channels
    if midas_img.ndim == 2:
        midas_img = np.expand_dims(midas_img, axis=-1)
        # stack
        midas_img = np.concatenate([midas_img, midas_img, midas_img], axis=-1)

    # adjust midas min value
    normalbae_img = value_map(normalbae_img, np.min(normalbae_img), np.max(normalbae_img), min_normalbae, 1.0)

    # adjust midas min value
    midas_img = value_map(midas_img, np.min(midas_img), np.max(midas_img), min_midas, 1.0)

    depth_scaler = normalbae_img * midas_img

    # normalize depth scaler
    depth_scaler = value_map(depth_scaler, np.min(depth_scaler), np.max(depth_scaler), min_depth_scale, 1.0)

    image = ade20k_img + openpose_img

    merged = image * depth_scaler
    merged = np.clip(merged, 0, 1) * 255
    merged = merged.astype(np.uint8)

    return [merged]


annotators.append(
    Annotator(
        name='Kitchen Sink',
        slug='kitchen_sink',
        call_override=kitchen_sink
    )
)


def post_process(annotated_img, original_image):
    img = annotated_img
    # is is list, get the first one
    if isinstance(img, list):
        img = img[0]
    img = HWC3(img)
    h, w, _ = original_image.shape
    ha, wa, _ = img.shape
    if h != ha or w != wa:
        output_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        output_img = img
    return output_img


def cleanup_annotators():
    for annotator in annotators:
        annotator.cleanup()


def annotate(input_image, args):
    res = args.res

    if res == -1:
        # get resolution
        orig_h, orig_w, orig_c = input_image.shape
        res = min(orig_h, orig_w)

    # clone numpy image
    img = input_image.copy()
    with torch.no_grad():
        # find the annotator
        for annotator_model in annotators:
            if annotator_model.slug == args.annotator:
                # build additional kwargs
                kwargs = {}
                if annotator_model.additional_args is not None:
                    for arg_dict in annotator_model.additional_args:
                        keyword = arg_dict['slug']
                        if 'keyword' in arg_dict:
                            keyword = arg_dict['keyword']
                        kwargs[keyword] = getattr(args, arg_dict['slug'])
                # run the model
                result = annotator_model(img, res, **kwargs)
                # post process
                result = post_process(result, img)
                return result

        # if we made it here, we didn't find the annotator
        raise Exception(f'Annotator {args.annotator} not found')
