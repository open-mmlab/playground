import json    # json模块是Python标准库中的一个模块，用于处理JSON格式的数据。在该代码中，使用json模块将检测结果转换为JSON格式。
from mmdet.apis import init_detector, inference_detector

import mmcv
# from mmdet_custom.datasets import D10007Dataset

def init():

    config_file = '/project/train/src_repo/mmyolo/tools/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py'
    checkpoint_file = '/project/train/models/train/exp/weights/epoch_320.pth'
    model = init_detector(config_file, checkpoint_file)
    # init_detector函数用于初始化MMDetection模型，输入参数包括模型配置文件路径config_file和模型权重文件路径checkpoint_file。该函数返回一个MMDetection模型对象model，用于后续的目标检测操作。
    return model

def process_image(handle=None, input_image=None, args=None, **kwargs):

    # CLASSES = D10007Dataset.CLASSES  # D10007Dataset是一个自定义的数据集类，用于加载数据集的类别信息。在该代码中，通过D10007Dataset.CLASSES获取数据集的类别信息，用于后续的处理。
    CLASSES = ['person', 'hat', 'head']
    # CLASSES 是一个 Python列表  例如CLASSES的值为["person", "car", "dog"]

    # result = inference_detector(handle, input_image)
    # # inference_detector函数用于对输入图像进行目标检测，输入参数包括MMDetection模型对象handle和输入图像input_image。
    # # 该函数返回一个包含检测结果的列表，其中每个元素表示一个检测框的信息，包括坐标、置信度、类别等。

    # labels = result.pred_instances.labels
    # objects = []
    # for i, class_name in enumerate(CLASSES):   # 遍历数据集的类别信息CLASSES，对每个类别的检测结果进行处理。
    #     # 具体来说，对于每个类别的检测结果，遍历其中的每个检测框，将检测框的 坐标、置信度、类别 等信息转换为字典格式，并添加到objects列表中。

    #     fires = result[i]
    #     for fire in fires:     # 具体来说，对于每个类别的检测结果，从result列表中获取该类别的检测结果fires，然后遍历其中的每个检测框fire。
    #         obj = dict(
    #             xmin = int(fire[0].item()),
    #             ymin= int(fire[1].item()),
    #             xmax=int(fire[2].item()),
    #             ymax=int(fire[3].item()),
    #             confidence=fire[4].item(),
    #             name = CLASSES[i]
    #         )
    #     # 对于每个检测框，将其坐标、置信度、类别等信息转换为字典格式，并添加到objects列表中。
    #     # 具体来说，使用dict函数创建一个字典对象obj，其中包含检测框的左上角和右下角坐标、置信度、类别等信息。然后，将obj添加到objects列表中。

    #         if obj['confidence' ] >0.5:
    #             objects.append(obj)

    result = inference_detector(handle, input_image)
    bboxes = result.pred_instances.bboxes
    scores = result.pred_instances.scores
    labels = result.pred_instances.labels
    objects = []
    fan = len(bboxes)
    for i in range(len(bboxes)):
            
            obj = dict(
            xmin = int(bboxes[i][0]),
            ymin = int(bboxes[i][1]),
            xmax = int(bboxes[i][2]),
            ymax = int(bboxes[i][3]),
            confidence = float(scores[i]),
            name = CLASSES[labels[i]])
            if obj['confidence'] > 0.5:
                            objects.append(obj)

    

    
    # model.show_result(img, result)
    # model.show_result(img, result, out_file='result.jpg', score_thr = 0.3)
    r_json = dict()
    r_json['algorithm_data'] = dict(target_info=objects, is_alert=False, target_count=0)
    r_json['model_data'] = dict(objects=objects)

    '''
    这段代码的作用是将MMDetection模型的检测结果转换为字典格式，并添加到objects列表中。具体分析如下：

    objects是一个空列表，用于存储检测结果的信息。

    在循环体中，使用enumerate函数遍历数据集的类别信息CLASSES，对每个类别的检测结果进行处理。具体来说，对于每个类别的检测结果，从result列表中获取该类别的检测结果fires，然后遍历其中的每个检测框fire。

    对于每个检测框，将其坐标、置信度、类别等信息转换为字典格式，并添加到objects列表中。具体来说，使用dict函数创建一个字典对象obj，
    其中包含检测框的左上角和右下角坐标、置信度、类别等信息。然后，将obj添加到objects列表中。

    在添加obj到objects列表之前，使用if语句判断检测框的置信度是否大于0.5。如果是，则将obj添加到objects列表中；否则，忽略该检测框。

    最后，将objects列表转换为字典格式，并添加到r_json字典中。其中，r_json字典包含两个键值对，分别为algorithm_data和model_data。
    algorithm_data表示算法的输出结果，包括目标信息、是否报警、目标数量等；model_data表示模型的输出结果，包括检测框的信息。
    '''

    if objects.__len__( ) >0:
        r_json['algorithm_data']['is_alert'] = True
        r_json['algorithm_data']['target_count'] = objects.__len__()

    # return json.dumps(objects, indent=4)
    return json.dumps(r_json, indent=4)

if __name__ == "__main__":
    handle = init()
    # 或者 img = mmcv.imread(img), 这将只加载图像一次．
    img = "/home/data/831/helmet_10809.jpg"

    process_image(handle, img, '{"mask_output_path": "result.png"}')