import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import groundingdino
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import get_tokenlizer
    from groundingdino.util.utils import clean_state_dict
    grounding_dino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
except ImportError:
    groundingdino = None

try:
    import mmcv
    import mmdet
    from mmdet.models.trackers import ByteTracker
    from mmdet.structures import DetDataSample
    from mmdet.visualization.local_visualizer import TrackLocalVisualizer
    from mmengine.config import Config
    from mmengine.structures import InstanceData
except ImportError:
    mmdet = None

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def parse_args():
    parser = argparse.ArgumentParser('Open Tracking Demo', add_help=True)
    parser.add_argument(
        '--config_file',
        '-c',
        type=str,
        required=True,
        help='path to config file')
    parser.add_argument(
        '--checkpoint_path',
        '-p',
        type=str,
        required=True,
        help='path to checkpoint file')
    parser.add_argument(
        '--inputs', '-i', type=str, help='path to video or image dirs')
    parser.add_argument('--text_prompt', '-t', type=str, help='text prompt')
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='outputs',
        help='output directory')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.05, help='box threshold')
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='running on cpu only!, default=False')

    # track params
    parser.add_argument(
        '--init_track_thr', type=float, default=0.4, help='init track')
    parser.add_argument(
        '--obj_score_thrs_high',
        type=float,
        default=0.3,
        help='first association threshold')
    parser.add_argument(
        '--obj_score_thrs_low',
        type=float,
        default=0.1,
        help='second association threshold')
    parser.add_argument(
        '--num_frames_retain',
        type=int,
        default=30,
        help='remove lost tracklet more than num frames')

    # video params
    parser.add_argument('--fps', type=int, default=30, help='video fps')
    parser.add_argument(
        '--out', type=str, default='demo.mp4', help='output video name')
    return parser.parse_args()


def build_grounding_dino_model(args):
    gdino_args = Config.fromfile(args.config_file)
    model = build_model(gdino_args)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, args):
    caption = args.text_prompt
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption = caption + '.'
    label_name = caption[:-1].split('. ')

    tokens_positive = []
    start_i = 0
    separation_tokens = '. '
    for _index, label in enumerate(label_name):
        end_i = start_i + len(label)
        tokens_positive.append([(start_i, end_i)])
        if _index != len(label_name) - 1:
            start_i = end_i + len(separation_tokens)
    tokenizer = get_tokenlizer.get_tokenlizer('bert-base-uncased')
    tokenized = tokenizer(caption, padding='longest', return_tensors='pt')
    positive_map_label_to_token = create_positive_dict(
        tokenized, tokens_positive, list(range(len(label_name))))

    device = 'cuda' if not args.cpu_only else 'cpu'
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)

    logits = convert_grounding_to_od_logits(
        logits, len(label_name),
        positive_map_label_to_token)  #[N, num_classes]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > args.box_thr
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    scores, pred_phrase_idx = logits_filt.max(1)

    return boxes_filt, pred_phrase_idx, scores, label_name


def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j, iff token i is mapped to j label"""

    positive_map_label_to_token = {}

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)

            assert beg_pos is not None and end_pos is not None
            positive_map_label_to_token[labels[j]] = []
            for i in range(beg_pos, end_pos + 1):
                positive_map_label_to_token[labels[j]].append(i)

    return positive_map_label_to_token


def convert_grounding_to_od_logits(logits,
                                   num_classes,
                                   positive_map,
                                   score_agg='MEAN'):
    """
    logits: (num_query, max_seq_len)
    num_classes: 80 for COCO
    """
    assert logits.ndim == 2
    assert positive_map is not None
    scores = torch.zeros(logits.shape[0], num_classes).to(logits.device)
    # 256 -> 80, average for each class
    # score aggregation method
    if score_agg == 'MEAN':  # True
        for label_j in positive_map:
            scores[:, label_j] = logits[:,
                                        torch.LongTensor(positive_map[label_j]
                                                         )].mean(-1)
    else:
        raise NotImplementedError
    return scores


def main():
    if groundingdino == None and mmdet == None:
        raise RuntimeError('detection model is not installed,\
                 please install it follow README')

    args = parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # define input
    if osp.isdir(args.inputs):
        imgs = sorted(
            filter(lambda x: x.endswith(IMG_EXTENSIONS),
                   os.listdir(args.inputs)),
            key=lambda x: x.split('.')[0])
        in_video = False
    else:
        imgs = []
        cap = cv2.VideoCapture(args.inputs)
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            imgs.append(frame)
        in_video = True

    # define fs
    fps = args.fps
    if args.show:
        if fps is None and in_video:
            fps = video_fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    tracker = ByteTracker(
        motion=dict(type='KalmanFilter'),
        obj_score_thrs=dict(
            high=args.obj_score_thrs_high, low=args.obj_score_thrs_low),
        init_track_thr=args.init_track_thr,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=args.num_frames_retain)

    visualizer = TrackLocalVisualizer()

    for frame_id, img in enumerate(imgs):
        if frame_id > 20:
            break
        if isinstance(img, str):
            image_path = osp.join(args.inputs, img)
            image_pil = Image.open(image_path).convert('RGB')
            image, _ = grounding_dino_transform(image_pil, None)
        else:
            image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image, _ = grounding_dino_transform(image_pil, None)

        out_file = os.path.join(args.out_dir, f'{frame_id:06d}.jpg')
        model = build_grounding_dino_model(args)
        boxes_filt, pred_phrases, scores_filt, label_name = get_grounding_output(
            model, image, args)

        visualizer.dataset_meta = {'classes': label_name}

        size = image_pil.size
        boxes_filt = boxes_filt * torch.tensor(
            [size[0], size[1], size[0], size[1]]).repeat(len(boxes_filt), 1)
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]

        # track input
        img_data_sample = DetDataSample()
        pred_instances = InstanceData()
        pred_instances.bboxes = boxes_filt
        pred_instances.labels = pred_phrases
        pred_instances.scores = scores_filt
        img_data_sample.pred_instances = pred_instances
        img_data_sample.set_metainfo(dict(frame_id=frame_id))

        pred_track_instances = tracker.track(img_data_sample)
        img_data_sample.pred_track_instances = pred_track_instances

        visualizer.add_datasample(
            'mot',
            np.asarray(image_pil),
            data_sample=img_data_sample,
            show=args.show,
            draw_gt=False,
            out_file=out_file,
            wait_time=float(1 / int(fps)) if fps else 0,
            pred_score_thr=0.0,
            step=frame_id)

    mmcv.frames2video(args.out_dir, args.out, fps=fps, fourcc='mp4v')


if __name__ == '__main__':
    main()
