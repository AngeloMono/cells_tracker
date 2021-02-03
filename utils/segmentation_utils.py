from yolact import Yolact
from yolact_utils.augmentations import FastBaseTransform
from data import cfg
from layers.output_utils import postprocess
import torch


def get_mask_bbox_and_score(yolact_net: Yolact, img, threshold=0.0, max_predictions=1):
    """
    Create and return the masks, bboxs and scores given the yolact net and the img
    :param yolact_net: Yolact net initialized
    :param img: ndarray img to segment
    :param threshold: threshold segmentation
    :param max_predictions: maximum number of predictions
    :returns:
        - masks_to_return - a list or single value of ndarray with 0 and 1 representing the mask(s)
        - boxes_to_return - a list or single value of ndarray representing the bbox(s)
        - scores_to_return - a list or single value of float in [0, 1] representing the confidence(s)
    """
    with torch.no_grad():
        frame = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = yolact_net(batch)

        h, w, _ = img.shape

        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True)
        cfg.rescore_bbox = save

        idx = t[1].argsort(0, descending=True)[:max_predictions]
        classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in t[:]]

        num_dets_to_consider = min(max_predictions, classes.shape[0])
        # Remove detections below the threshold
        for j in range(num_dets_to_consider):
            if scores[j] < threshold:
                num_dets_to_consider = j
                break
        masks_to_return = boxes_to_return = scores_to_return = None
        if num_dets_to_consider > 0:
            masks = masks[:num_dets_to_consider, :, :, None]
            masks_to_return = []
            boxes_to_return = []
            scores_to_return = []
            for m, b, s in zip(masks, boxes, scores):
                masks_to_return.append(m)
                boxes_to_return.append(b)
                scores_to_return.append(s)
            if len(masks_to_return) == 1:
                masks_to_return = masks_to_return[0]
            if len(boxes_to_return) == 1:
                boxes_to_return = boxes_to_return[0]
            if len(scores_to_return) == 1:
                scores_to_return = scores_to_return[0]
        return masks_to_return, boxes_to_return, scores_to_return
