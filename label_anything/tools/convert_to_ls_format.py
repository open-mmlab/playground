import argparse
import json
import os
import uuid
try:
    from label_studio_converter.imports.label_config import generate_label_config
except:
    raise ModuleNotFoundError(
        "label_studio_converter is not installed, run `pip install label_studio_converter` to install")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO labeling to Label Studio JSON")
    parser.add_argument('--input-file', help='JSON file with COCO annotations')
    parser.add_argument('--output-file', help='output Label Studio JSON file')
    parser.add_argument(
        '--image-root-url', help='root URL path where images will be hosted, e.g.: http://example.com/images', default='/data/local-files/?d=')

    args = parser.parse_args()
    return args


def new_task(out_type, root_url, file_name):
    """create new task with Label Studio format
    copy from: https://github.com/heartexlabs/label-studio-converter/blob/master/label_studio_converter/imports/coco.py

    Args:
        out_type (str): labeling out_type in Label Studio.
        root_url (str): image root_url.
        file_name (str): image file_name.

    Returns:
        dict: task info dict 
    """
    return {
        "data": {"image": os.path.join(root_url, file_name)},
        # 'annotations' or 'predictions'
        out_type: [
            {
                "result": [],
                "ground_truth": False,
            }
        ],
    }


def create_bbox(annotation, categories, from_name, image_height, image_width, to_name):
    """create bbox labeling with Label Studio format.
    copy from: https://github.com/heartexlabs/label-studio-converter/blob/master/label_studio_converter/imports/coco.py

    Args:
        annotation (dict): annotation dict with COCO format.
        categories (List): a list of categories.
        from_name (str): Name of the tag used to label the region in Label Studio.
        image_height (int): height of image.
        image_width (int): width of image.
        to_name (str): Name of the object tag that provided the region to be labeled.

    Returns:
        dict: an labeling dict with Label Studio format.
    """
    label = categories[int(annotation['category_id'])]
    x, y, width, height = annotation['bbox']
    x, y, width, height = float(x), float(y), float(width), float(height)
    item = {
        "id": uuid.uuid4().hex[0:10],
        "type": "rectanglelabels",
        "value": {
            "x": x / image_width * 100.0,
            "y": y / image_height * 100.0,
            "width": width / image_width * 100.0,
            "height": height / image_height * 100.0,
            "rotation": 0,
            "rectanglelabels": [label],
        },
        "to_name": to_name,
        "from_name": from_name,
        "image_rotation": 0,
        "original_width": image_width,
        "original_height": image_height,
    }
    return item


def convert_coco_to_ls(input_file,
                       out_file,
                       from_name='label',
                       to_name='image',
                       out_type="annotations",
                       use_super_categories=False,
                       image_root_url='/data/local-files/?d=',):
    """Convert COCO labeling to Label Studio JSON
    Modified from: https://github.com/heartexlabs/label-studio-converter/blob/master/label_studio_converter/imports/coco.py

    Args:
        input_file (str): input json file path with COCO annotations
        out_file (str): output json file path with Label Studio annotations
        from_name (str, optional): Name of the tag used to label the region in Label Studio. Defaults to 'label'.
        to_name (str, optional): Name of the object tag that provided the region to be labeled. Defaults to 'image'.
        use_super_categories (bool, optional): whether to use super_categories in COCO. Defaults to False.
        image_root_url (str, optional): Image path prefix. Defaults to '/data/local-files/?d='.
    """
    tasks = {}  # image_id => task
    print(f'Reading COCO notes and categories from', input_file)
    with open(input_file, encoding='utf8') as f:
        coco = json.load(f)

    # build categories => labels dict
    new_categories = {}
    # list to dict conversion: [...] => {category_id: category_item}
    categories = {int(category['id'])                  : category for category in coco['categories']}
    ids = sorted(categories.keys())  # sort labels by their origin ids

    for i in ids:
        name = categories[i]['name']
        if use_super_categories and 'supercategory' in categories[i]:
            name = categories[i]['supercategory'] + ':' + name
        new_categories[i] = name

    # mapping: id => category name
    categories = new_categories

    # mapping: image id => image
    images = {item['id']: item for item in coco['images']}

    print(
        f'Found {len(categories)} categories, {len(images)} images and {len(coco["annotations"])} annotations')

    # flags for labeling config composing
    bbox = False
    bbox_once = False
    rectangles_from_name = from_name + '_rectangles'
    tags = {}

    # create tasks
    for image in coco['images']:
        image_id, image_file_name = image['id'], image['file_name']
        tasks[image_id] = new_task(out_type, image_root_url, image_file_name)

    for i, annotation in enumerate(coco['annotations']):
        bbox |= 'bbox' in annotation  # if bbox
        if bbox and not bbox_once:
            tags.update({rectangles_from_name: 'RectangleLabels'})
            bbox_once = True

        # read image sizes
        image_id = annotation['image_id']
        image = images[image_id]
        image_file_name, image_width, image_height = (
            image['file_name'],
            image['width'],
            image['height'],
        )
        task = tasks[image_id]

        if 'bbox' in annotation:
            item = create_bbox(
                annotation,
                categories,
                rectangles_from_name,
                image_height,
                image_width,
                to_name,
            )
            task[out_type][0]['result'].append(item)

        tasks[image_id] = task

    # generate and save labeling config
    label_config_file = out_file.replace('.json', '') + '.label_config.xml'
    generate_label_config(categories, tags, to_name,
                          from_name, label_config_file)

    if len(tasks) > 0:
        tasks = [tasks[key] for key in sorted(tasks.keys())]
        print('Saving Label Studio JSON to', out_file)
        with open(out_file, 'w') as out:
            json.dump(tasks, out)

        print(
            '\n'
            f'Following the instructions to load data into Label Studio\n'
            f'  1. Create a new project in Label Studio\n'
            f'  2. Use Labeling Config from "{label_config_file}"\n'
            f'  3. Setup serving for images [e.g. you can use Local Storage (or others):\n'
            f'     https://labelstud.io/guide/storage.html#Local-storage]\n'
            f'  4. Import "{out_file}" to the project\n'
        )
    else:
        print('No labels converted')


if __name__ == '__main__':
    args = parse_args()
    convert_coco_to_ls(
        input_file=args.input_file,
        out_file=args.output_file,
        image_root_url=args.image_root_url)
