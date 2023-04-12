from mmcv.transforms import BaseTransform
from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AddConvertedGTBox(BaseTransform):
    """Convert boxes in results to a certain box type."""

    def __init__(self, box_type_mapping: dict) -> None:
        self.box_type_mapping = box_type_mapping

    def transform(self, results: dict) -> dict:
        """The transform function."""
        for key, dst_box_type in self.box_type_mapping.items():
            assert key != 'gt_bboxes'
            gt_bboxes = results['gt_bboxes']
            results[key] = gt_bboxes.convert_to(dst_box_type)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(box_type_mapping={self.box_type_mapping})'
        return repr_str
