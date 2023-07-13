## Batch Annotator

This is a simple tool to annotate images in batches using various models.

### Installation

```bash
git submodule update --init --recursive
pip install -r requirements.txt
``` 

### Usage

```bash
python main.py <input_dir> <output_dir> <annotator>
```

### Annotators

| Annotator        | Image                                                                                                                           |
|------------------|---------------------------------------------------------------------------------------------------------------------------------|
|                  | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img.jpg" width="384" height="480">               |
| canny            | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_canny.jpg" width="384" height="480">         |
| color_shuffler   | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_color_shuffler.jpg" width="384" height="480"> |
| content_shuffler | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_content_shuffler.jpg" width="384" height="480"> |
| hed              | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_hed.jpg" width="384" height="480">           |
| lineart          | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_lineart.jpg" width="384" height="480">       |
| lineart_anime    | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_lineart_anime.jpg" width="384" height="480"> |
| midas            | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_midas.jpg" width="384" height="480">         |
| mlsd             | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_mlsd.jpg" width="384" height="480">          |
| normalbae        | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_normalbae.jpg" width="384" height="480">     |
| oneformer_ade20k | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_oneformer_ade20k.jpg" width="384" height="480"> |
| oneformer_coco   | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_oneformer_coco.jpg" width="384" height="480"> |
| openpose         | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_openpose.jpg" width="384" height="480">      |
| pidi             | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_pidi.jpg" width="384" height="480">          |
| uniformer        | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_uniformer.jpg" width="384" height="480">     |
| zoe              | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_zoe.jpg" width="384" height="480">           |
| midas_ade20k              | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_midas_ade20k.jpg" width="384" height="480">              |
| normalbae_ade20k              | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_normalbae_ade20k.jpg" width="384" height="480">              |
| kitchen_sink              | <img src="https://raw.githubusercontent.com/ostris/batch-annotator/main/assets/img_kitchen_sink.jpg" width="384" height="480">              |

