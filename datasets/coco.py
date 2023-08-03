# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T

'''
    img: 이는 하나의 이미지를 나타낸다. COCO 데이터셋에서의 이미지는 일반적으로
    RGB 포맷의 PIL 이미지 또는 numpy array
    
    target: 이미지에 대한 주석(annotation).

'''
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        # transform을 해주어야 한다면. 해준다.
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


'''
    COCO 데이터셋의 세그멘테이션 정보를 마스크 형태로 변환하는 역할을 한다.
    
    함수의 입력으로 segmentations는 COCO 데이터셋에서 특정 이미지의 모든 객체에 대한 폴리곤(polygon) 좌표를 담고 있는 리스트이다.
    그리고 height와 width는 해당 이미지의 높이와 너비이다.
    
    polygon은 [x1, y1, x2, y2,..., xn, yn]이런 식으로 나타내진다. 폴리곤 세그멘테이션은 일반적으로 대상 객체의 픽셀 수준에서의 정확한 위치와 형태
    이를 COCO에 적용하기 위해서는 RLE(Run-Length Encoding)을 적용해주어야 하는데, 이진 마스크로 인코딩 해야 한다. 
    이렇게 해주는 이유는 대부분의 딥러닝 모델이 픽셀 수준의 이진 마스크 형태의 입력을 필요로 하기 때문이다. 그래야 모델에게 명확한 위치 정보와 객체의 형태를 전달 할 수 있게된다.
'''
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width) # 각 폴리곤을 이진 마스크로 변환한다. RLE 형식으로 변환
        mask = coco_mask.decode(rles) # 
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

'''
    COCO 데이터셋 형식의 이미지와 대상(annotation)을 받아서 처리하여 적절한 형태로 변환하는 것이다.
    이 과정에서 bounding box, class label, mask, keypoints등이 추출되고, 특정 조건에 따라 정리
    
'''
class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        # target의 어노테이션 정보를 불러온다.
        anno = target["annotations"]

        # iscrowd가 0이거나 없는 경우만 필터링해온다.
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # bbox 값을 가져온다.
        boxes = [obj["bbox"] for obj in anno]
        
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# image를 빌드해온다. 기본은 coco 데이터셋
def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    # segmentation이면 mask를 제공해주어야 한다.
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
