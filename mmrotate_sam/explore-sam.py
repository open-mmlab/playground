# use `torch.save(dict(img=img, h_bboxes=h_bboxes), 'tmp0.pth')` get data
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from utils import show_mask, show_points, show_box


# INPUT_BOXES = True
# INPUT_POINTS = False


def main(data_path, sam_checkpoint, model_type, INPUT_BOXES, INPUT_POINTS):
    data = torch.load(data_path)
    img = data['img']
    h_bboxes = data.get('h_bboxes', data['h_boxes'])
    x1y2x2y1 = h_bboxes[:, (0, 3, 2, 1)]
    center_points = ((h_bboxes[:, :2] + h_bboxes[:, 2:]) / 2).unsqueeze(1)
    corner_points = torch.stack(
        [h_bboxes[:, :2], h_bboxes[:, 2:],
         x1y2x2y1[:, :2], x1y2x2y1[:, 2:]], dim=1)
    center_labels = torch.ones((center_points.shape[:2]),
                               device=center_points.device)
    # TODO: Implement Corner position of RBox
    corner_labels = torch.zeros((corner_points.shape[:2]),
                                device=corner_points.device)
    points = torch.cat([center_points, corner_points], dim=1)
    labels = torch.cat([center_labels, corner_labels], dim=1)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    transformed_points = predictor.transform.apply_coords_torch(points,
                                                                img.shape[:2])
    transformed_bboxes = predictor.transform.apply_boxes_torch(h_bboxes,
                                                               img.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=transformed_points if INPUT_POINTS else None,
        point_labels=labels if INPUT_POINTS else None,
        boxes=transformed_bboxes if INPUT_BOXES else None,
        multimask_output=False)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    if INPUT_BOXES:
        for box in h_bboxes:
            show_box(box, plt.gca())
    if INPUT_POINTS:
        for _points, _labels in zip(points, labels):
            show_points(_points, _labels, plt.gca(), 150)
    plt.axis('off')
    flag = lambda f: 't' if f else 'f'
    plt.savefig(
        f'{data_path[:-4]}_box-{flag(INPUT_BOXES)}_point={flag(INPUT_POINTS)}.png')


if __name__ == '__main__':
    data_path_list = [f'tmp{i}.pth' for i in range(3)]
    sam_checkpoint = r"../segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    for data_path in data_path_list:
        for a in (True, False):
            for b in (True, False):
                main(data_path, sam_checkpoint, model_type, a, b)
