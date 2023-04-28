import json
import os

coco_format = {}
images = []
categories = []
annotations = []
bbox = []
area = []

def format_to_coco(json_file_path):
    
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
    for index_annotation in range(len(contents)):
        result = contents[index_annotation]['annotations'][0]['was_cancelled']

        if result == False:
            width_from_json = contents[index_annotation]["annotations"][0]["result"][0]["original_width"]
            height_from_json = contents[index_annotation]["annotations"][0]["result"][0]["original_height"]
            image_id = contents[index_annotation]["id"]
            image_json_name = contents[index_annotation]["data"]['image'].split('/')[-1]
            image_json_name = image_json_name.split('-', 1)[1]
            category = contents[index_annotation]['annotations'][0]['result'][0]['value']['rectanglelabels'][0]
            organized_pixels_coordenates = contents[index_annotation]['annotations'][0]['result'][1]['value']['rle']
            label = contents[index_annotation]['annotations'][0]['result']
            bbox = [label[0]['value']['x'], label[0]['value']['y'], label[0]['value']['width'], label[0]['value']['height']]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

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
                "segmentation": [organized_pixels_coordenates],
                "bbox": bbox,
                "area": area,
                'iscrowd': 0#rle format
                
            })

    # 将coco格式保存到文件中
    output_dir="coco_format_files/"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"coco_format_files/coco_file.json", "w") as out_file:
        json.dump(coco_format, out_file, ensure_ascii=False, indent=4)


def saving_json(list_data):
    output_dir="coco_format_files/"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"coco_format_files/coco_file.json", "w") as out_file:
                json.dump(list_data, out_file, ensure_ascii=False, indent=4)

def set_categories(categories):

    set_categories = []
    list_data = []

    for index in range(len(categories)):
        set_categories.append(categories[index]['name'])

    index_set = sorted(list(set(set_categories)))

    for index in range(len(index_set)):
        data = {'id': index, 'name': index_set[index]}  
        list_data.append(data)

    return list_data

def organizing_coordenates(pixels_coordenates):
    x_y_coordinates = []

    for index in range(0, len(pixels_coordenates)):
        x_y_coordinates.append(pixels_coordenates[index][0])
        x_y_coordinates.append(pixels_coordenates[index][1])
        
    return x_y_coordinates


def total_area_bounding_box(pixels_coordenates):
    x_coordinates = []
    y_coordinates = []
    result = []

    for index in range(len(pixels_coordenates)):

        x_coordinates.append(pixels_coordenates[index][0])
        y_coordinates.append(pixels_coordenates[index][1])

    x_min, y_min, x_max, y_max = (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))
    area_segmentation = (x_max - x_min) * (y_max - y_min)
    result.append([x_min, y_min, x_max, y_max, area_segmentation])
    
    return result
 
format_to_coco('label_studio.json')

    
