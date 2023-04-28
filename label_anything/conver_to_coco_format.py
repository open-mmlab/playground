import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from itertools import groupby
from label_studio_converter.brush import decode_rle

def parse_args():
    parser = argparse.ArgumentParser(description='Label studio convert to Coco fomat')
    parser.add_argument('--json_file_path',default='label_studio.json', help='label studio output json')
    parser.add_argument('--output_dir',default='coco_format_files', help='output dir of Coco format json')

    args = parser.parse_args()
    return args


coco_format = {}
images = []
categories = []
annotations = []
bbox = []
area = []

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def rle2mask(rle,height, width):
    """Convert RLE to mask

    :param rle: list of ints in RLE format
    :param height: height of binary mask
    :param width: width of binary mask
    :return: binary mask
    """
    mask = decode_rle(rle)
    mask = mask.reshape(-1,4)[:, 0]
    if mask.max()==255:
        mask=np.where(mask==255,1,0)

    # mask = mask[:len(mask)//4]

    return mask.reshape((height, width))

def format_to_coco(args):
    json_file_path=args.json_file_path
    
    # 初始化coco格式的字典
    coco_format = {
        "images": [],
        "categories": [],
        "annotations": []
    }

    # 读取label studio格式的JSON文件
    with open(json_file_path, 'r') as file:
        contents = json.loads(file.read())

    # 遍历每个标注
    for index_annotation in tqdm(range(len(contents))):
        result = contents[index_annotation]['annotations'][0]['was_cancelled']

        if result == False:
            width_from_json = contents[index_annotation]["annotations"][0]["result"][0]["original_width"]
            height_from_json = contents[index_annotation]["annotations"][0]["result"][0]["original_height"]

            label = contents[index_annotation]['annotations'][0]['result']

            x = label[0]['value']['x'] * width_from_json / 100
            y = label[0]['value']['y'] * height_from_json / 100
            w = label[0]['value']['width'] * width_from_json / 100
            h = label[0]['value']['height'] * height_from_json / 100

            image_id = contents[index_annotation]["id"]
            image_json_name = contents[index_annotation]["data"]['image'].split('/')[-1]
            image_json_name = image_json_name.split('-', 1)[1]
            category = contents[index_annotation]['annotations'][0]['result'][0]['value']['rectanglelabels'][0]
            rle = contents[index_annotation]['annotations'][0]['result'][1]['value']['rle']
            mask=rle2mask(rle,height_from_json,width_from_json)
            rle=binary_mask_to_rle(mask)
            bbox = [x, y, w, h]
            area = w * h

            # 添加图像信息到coco格式
            coco_format["images"].append({
                "id": image_id,
                "file_name": image_json_name,
                "width": width_from_json,
                "height": height_from_json
            })

            # 检查类别是否已经添加到categories变量中，如果没有，将其添加到categories变量
            if not any(d["name"] == category for d in coco_format["categories"]):
                coco_format["categories"].append({
                    "id": len(coco_format["categories"]),
                    "name": category
                })

            # 添加标注信息到coco格式
            coco_format["annotations"].append({
                "id": index_annotation,
                "image_id": image_id,
                "category_id": [d["id"] for d in coco_format["categories"] if d["name"] == category][0],
                "segmentation": rle,
                "bbox": bbox,
                "area": area,
                'iscrowd': 0#rle format
                
            })

    # 将coco格式保存到文件中
    output_dir=args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir,'output.json'), "w") as out_file:
        json.dump(coco_format, out_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    args = parse_args()
    format_to_coco(args)
   
