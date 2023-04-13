import json
import argparse
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import mmengine

def parse_args():
    parser = argparse.ArgumentParser(description='Convert the word polygon to line polygon')
    parser.add_argument(
        '--json_file', 
        default='/home/jq/ICDAR-2023-HierText/annotation/det_annotations/word_level/train.json',
        help='The json file needed to convert'
    )
    parser.add_argument(
        '--out_file', 
        default='/home/jq/ICDAR-2023-HierText/annotation/det_annotations/merge_polygon/para_level/train.json',
        help='The json file needed to convert'
    )
    parser.add_argument(
        '--mode',
        default='para',
        help='The mode for word polygon to whether line polygon or paragraph polygon'
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    with open(args.json_file, 'r') as f:
        json_data = json.load(f)
    
    out_dict = {}
    out_dict['metainfo'] = json_data['metainfo']
    out_dict['data_list'] = []
    all_image_infos = {}
    for i, info in enumerate(json_data['data_list']):
        all_image_infos[i] = {}
        if args.mode == 'line':
            for instance in info['instances']:
                line_id = instance['line_id']
                if line_id in all_image_infos[i].keys():
                    all_image_infos[i][line_id].append(instance)
                else:
                    all_image_infos[i][line_id] = [instance]
        elif args.mode == 'para':
            for instance in info['instances']:
                para_id = instance['para_id']
                if para_id in all_image_infos[i].keys():
                    all_image_infos[i][para_id].append(instance)
                else:
                    all_image_infos[i][para_id] = [instance]
    
    for i, img_id in enumerate(all_image_infos.keys()):
        img_infos = all_image_infos[img_id]

        new_img_infos = {}
        new_img_infos['img_path'] = json_data['data_list'][i]['img_path']
        new_img_infos['height'] = json_data['data_list'][i]['height']
        new_img_infos['width'] = json_data['data_list'][i]['width']
        new_instances = []
        for l_p_id in img_infos.keys():
            new_instance = {}
            new_instance['bbox_label'] = 0
            l_p_info = img_infos[l_p_id]
            new_instance['ignore'] = l_p_info[0]['ignore']
            polygons = []
            for l_p in l_p_info:
                poly = l_p['polygon']
                polygons.append(poly)
            
            # Create a list of polygons
            l_p_polygons = []
            for p in polygons:
                p_x = [p[x_index] for x_index in range(0, len(p), 2)]
                p_y = [p[y_index] for y_index in range(1, len(p), 2)]
                poly_points = [(x, y) for x, y in zip(p_x, p_y)]
                l_p_polygons.append(Polygon(poly_points))

            # Merge the polygons into a single polygon
            merged_polygon = unary_union(l_p_polygons)
            poly_points = []
            if merged_polygon.geom_type == 'MultiPolygon':
                p_list = list(merged_polygon.convex_hull.exterior.coords)
            else:
                p_list = list(merged_polygon.exterior.coords)
            for (x, y) in p_list:
                poly_points.append(x)
                poly_points.append(y)
            new_instance['polygon'] = poly_points
            new_instances.append(new_instance)
        
        new_img_infos['instances'] = new_instances
        out_dict['data_list'].append(new_img_infos)
        print(f"Finished the image {i}")
    
    mmengine.dump(out_dict, args.out_file)
    print()