import numpy as np
import cv2
import onnxruntime

coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class FREEYOLOONNX(object):
    def __init__(
        self,
        model,
        img_size=640,
        score_thr=0.35

    ):
        self.model = model
        self.img_size = img_size
        self.score_thr = score_thr

        # onnx session
        self.session = onnxruntime.InferenceSession(self.model)

        # class color for better visualization
        np.random.seed(0)
        self.class_colors = [(np.random.randint(255),
                        np.random.randint(255),
                        np.random.randint(255)) for _ in range(80)]

        # preprocessor
        self.prepocess = PreProcessor(img_size=self.img_size)

        # postprocessor
        self.postprocess = PostProcessor(
            img_size=self.img_size, strides=[8, 16, 32],
            num_classes=80, conf_thresh=self.score_thr, nms_thresh=0.5)
    
    def infer(self, origin_img):
        # preprocess
        x, ratio = self.prepocess(origin_img)
   
        # inference
        ort_inputs = {self.session.get_inputs()[0].name: x[None, :, :, :]}
        output = self.session.run(None, ort_inputs)

        # post process
        bboxes, scores, labels = self.postprocess(output[0])
        bboxes /= ratio

        # visualize detection
        origin_img = self.visualize(
            img=origin_img,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            vis_thresh=self.score_thr,
            class_colors=self.class_colors
            )
        
        return origin_img
    
    def plot_bbox_labels(self, img, bbox, label, cls_color, test_scale=0.4):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        # plot bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * test_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, test_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        return img

    def visualize(self, img, bboxes, scores, labels, class_colors, vis_thresh=0.3):
        ts = 0.4
        for i, bbox in enumerate(bboxes):
            if scores[i] > vis_thresh:
                cls_color = class_colors[int(labels[i])]
                cls_id = coco_class_index[int(labels[i])]
                mess = '%s: %.2f' % (coco_class_labels[cls_id], scores[i])
                img = self.plot_bbox_labels(img, bbox, mess, cls_color, test_scale=ts)

        return img


class PostProcessor(object):
    def __init__(self, img_size, strides, num_classes, conf_thresh=0.15, nms_thresh=0.5):
        self.img_size = img_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = strides

        # generate anchors
        self.anchors, self.expand_strides = self.generate_anchors()


    def generate_anchors(self):
        """
            fmp_size: (List) [H, W]
        """
        all_anchors = []
        all_expand_strides = []
        for stride in self.strides:
            # generate grid cells
            fmp_h, fmp_w = self.img_size // stride, self.img_size // stride
            anchor_x, anchor_y = np.meshgrid(np.arange(fmp_w), np.arange(fmp_h))
            # [H, W, 2]
            anchor_xy = np.stack([anchor_x, anchor_y], axis=-1)
            shape = anchor_xy.shape[:2]
            # [H, W, 2] -> [HW, 2]
            anchor_xy = (anchor_xy.reshape(-1, 2) + 0.5) * stride
            all_anchors.append(anchor_xy)

            # expanded stride
            strides = np.full((*shape, 1), stride)
            all_expand_strides.append(strides.reshape(-1, 1))

        anchors = np.concatenate(all_anchors, axis=0)
        expand_strides = np.concatenate(all_expand_strides, axis=0)

        return anchors, expand_strides


    def decode_boxes(self, anchors, pred_regs):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors[..., :2] + pred_regs[..., :2] * self.expand_strides
        # size of bbox
        pred_box_wh = np.exp(pred_regs[..., 2:]) * self.expand_strides

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)

        return pred_box


    def __call__(self, predictions):
        """
        Input:
            predictions: (ndarray) [n_anchors_all, 4+1+C]
        """
        reg_preds = predictions[..., :4]
        obj_preds = predictions[..., 4:5]
        cls_preds = predictions[..., 5:]
        scores = np.sqrt(obj_preds * cls_preds)

        # scores & labels
        labels = np.argmax(scores, axis=1)                      # [M,]
        scores = scores[(np.arange(scores.shape[0]), labels)]   # [M,]

        # bboxes
        bboxes = self.decode_boxes(self.anchors, reg_preds)     # [M, 4]    

        # thresh
        keep = np.where(scores > self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        scores, labels, bboxes = self.multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, True)

        return bboxes, scores, labels

    def nms(self, bboxes, scores, nms_thresh):
        """"Pure Python NMS."""
        x1 = bboxes[:, 0]  #xmin
        y1 = bboxes[:, 1]  #ymin
        x2 = bboxes[:, 2]  #xmax
        y2 = bboxes[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(iou <= nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms_class_agnostic(self, scores, labels, bboxes, nms_thresh):
        # nms
        keep = self.nms(bboxes, scores, nms_thresh)

        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        return scores, labels, bboxes

    def multiclass_nms_class_aware(self, scores, labels, bboxes, nms_thresh, num_classes):
        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        return scores, labels, bboxes

    def multiclass_nms(self, scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
        if class_agnostic:
            return self.multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
        else:
            return self.multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)


class PreProcessor(object):
    def __init__(self, img_size):
        self.img_size = img_size
        self.input_size = [img_size, img_size]
        

    def __call__(self, image, swap=(2, 0, 1)):
        """
        Input:
            image: (ndarray) [H, W, 3] or [H, W]
            formar: color format
        """
        if len(image.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), np.float32) * 114.
        else:
            padded_img = np.ones(self.input_size, np.float32) * 114.
        # resize
        orig_h, orig_w = image.shape[:2]
        r = min(self.input_size[0] / orig_h, self.input_size[1] / orig_w)
        resize_size = (int(orig_w * r), int(orig_h * r))
        if r != 1:
            resized_img = cv2.resize(image, resize_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = image

        # padding
        padded_img[:resized_img.shape[0], :resized_img.shape[1]] = resized_img
        
        # [H, W, C] -> [C, H, W]
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)


        return padded_img, r