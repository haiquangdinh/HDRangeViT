### Evaluation
import torch
import torch.nn.functional as F
def compute_iou(preds, labels, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ious = []
    correct = (preds == labels)
    accuracy = correct.sum().float() / labels.numel()

    for cls in range(num_classes):
        # Get binary predictions and labels for this class
        pred_cls = (preds == cls)
        label_cls = (labels == cls)

        # Intersection and Union
        intersection = (pred_cls & label_cls).sum().float()
        union = (pred_cls | label_cls).sum().float()

        if union == 0:
            ious.append(torch.tensor(float('nan'), device=device))  # undefined for this class
        else:
            ious.append(intersection / union)

    # Mean IoU (excluding NaNs)
    ious_tensor = torch.stack(ious)
    mIoU = torch.nanmean(ious_tensor)

    return mIoU, ious_tensor, accuracy