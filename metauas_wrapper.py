# add
import numpy as np
import pytorch_lightning as pl
from loguru import logger as L
from pytorch_lightning.utilities import rank_zero_only
from data.datamodule import DataModule
import utils
from detectron2.structures.image_list import ImageList
from sklearn.metrics import average_precision_score
import torch
from torchvision.utils import make_grid
from torch.nn import functional as F

from metauas import MetaUAS


def calculate_iou_dice(pred, target, threshold=0.5, smooth=1e-6):
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred)
    if not torch.is_tensor(target):
        target = torch.tensor(target)

    if pred.min() < 0 or pred.max() > 1:
        pred = pred.sigmoid()

    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    batch_size = pred_binary.size(0)
    iou_list = []
    dice_list = []

    for i in range(batch_size):
        pred_flat = pred_binary[i].view(-1)
        target_flat = target_binary[i].view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou.item())

        dice = (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_list.append(dice.item())

    return iou_list, dice_list


def calculate_ap(pred, target, threshold=0.5, smooth=1e-6):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)

    B, C, H, W = pred.shape

    if target.dim() == 3:
        target_one_hot = torch.zeros(B, C, H, W)
        for i in range(B):
            target_one_hot[i] = F.one_hot(target[i].long(), num_classes=C).permute(2, 0, 1)
        target = target_one_hot

    if pred.min() < 0 or pred.max() > 1:
        pred = pred.sigmoid()

    ap_list = []

    for i in range(B):
        sample_aps = []

        for c in range(C):
            pred_flat = pred[i, c].flatten().detach().cpu().numpy()
            target_flat = target[i, c].flatten().detach().cpu().numpy()

            target_binary = (target_flat > threshold).astype(np.float32)

            if np.sum(target_binary) == 0:
                continue

            try:
                ap = average_precision_score(target_binary, pred_flat)
                sample_aps.append(ap)
            except:
                ap_manual = calculate_ap_manual(pred_flat, target_binary, smooth)
                sample_aps.append(ap_manual)

        if len(sample_aps) > 0:
            sample_ap = np.mean(sample_aps)
        else:
            sample_ap = 0.0

        ap_list.append(sample_ap)

    return ap_list


def visualize(prompt_image, query_image, mask_preds, gt,
              threshold=0.1, nrow=2, normalize_preds=True
              ):
    with torch.no_grad():
        batch_size = prompt_image.shape[0]

        def process_preds(preds):
            if normalize_preds and (preds.min() < 0 or preds.max() > 1):
                preds_normalized = torch.sigmoid(preds)
            else:
                preds_normalized = torch.clamp(preds, 0, 1)

            preds_binary = (preds_normalized > threshold).float()
            return preds_normalized, preds_binary

        def min_max_norm(image):
            return (image - image.min()) / (image.max() - image.min())

        preds_norm, preds_bin = process_preds(mask_preds)

        for idx in range(len(prompt_image)):
            prompt_image[idx] = min_max_norm(prompt_image[idx])
        for idx in range(len(query_image)):
            query_image[idx] = min_max_norm(query_image[idx])

        images_to_display = []

        for i in range(min(batch_size, 4)):
            images_to_display.extend([
                prompt_image[i],
                query_image[i],
                preds_norm[i].repeat(3, 1, 1),
                preds_bin[i].repeat(3, 1, 1),
                gt[i].repeat(3, 1, 1)
            ])

        if images_to_display:

            grid = make_grid(images_to_display,
                             nrow=nrow * 5, padding=10, pad_value=0.8)
            return grid
        else:
            raise ValueError("No Valid Image To Show")


class MetaUASWrapper(pl.LightningModule):
    def __init__(self, args,
                 encoder_name=None,
                 decoder_name=None,
                 encoder_depth=None,
                 decoder_depth=None,
                 num_alignment_layers=None,
                 alignment_type=None,
                 fusion_policy=None):
        super().__init__()
        self.args = args
        self.model = MetaUAS(
            encoder_name=encoder_name if encoder_name else args.encoder_name,
            decoder_name=decoder_name if decoder_name else args.decoder_name,
            encoder_depth=encoder_depth if encoder_depth else args.encoder_depth,
            decoder_depth=decoder_depth if decoder_depth else args.decoder_depth,
            num_alignment_layers=num_alignment_layers if num_alignment_layers else args.num_alignment_layers,
            alignment_type=alignment_type if alignment_type else args.alignment_type,
            fusion_policy=fusion_policy if fusion_policy else args.fusion_policy
        )

        self.test_set_names = [test_set.name for test_set in args.datasets.test_datasets]
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.loss = F.binary_cross_entropy
        if args.load_weights_from is not None:
            self.safely_load_state_dict(torch.load(args.load_weights_from, weights_only=False)["state_dict"])

        self.save_hyperparameters()
        self.validation_step_outputs = []

    def safely_load_state_dict(self, checkpoint_state_dict):
        model_state_dict = self.state_dict()
        for k in checkpoint_state_dict:
            if k in model_state_dict:
                if checkpoint_state_dict[k].shape != model_state_dict[k].shape:
                    L.log(
                        "INFO",
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {checkpoint_state_dict[k].shape}",
                    )
                    checkpoint_state_dict[k] = model_state_dict[k]
            else:
                L.log("INFO", f"Dropping parameter {k}")
        self.load_state_dict(checkpoint_state_dict, strict=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CenterNetWithCoAttention")
        parser.add_argument("--lr", type=float)
        parser.add_argument("--weight_decay", type=float)
        parser.add_argument("--encoder", type=str, choices=["resnet50", "resnet18"])
        parser.add_argument("--coam_layer_data", nargs="+", type=int)
        parser.add_argument("--attention", type=str)
        parser.add_argument("--decoder_attention", type=str, default=None)
        return parent_parser

    def training_step(self, batch, batch_idx):
        mask_preds = self(batch)
        prompt_image_masks, query_image_masks = batch["prompt_image_target_masks"], batch["query_image_target_masks"]
        overall_loss = self.loss(mask_preds, query_image_masks)

        self.log("train/loss", overall_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return overall_loss

    def validation_step(self, batch, batch_index):
        with torch.no_grad():
            mask_preds = self(batch)
            prompt_image_masks, query_image_masks = batch["prompt_image_target_masks"], batch[
                "query_image_target_masks"]
            overall_loss = self.loss(mask_preds, query_image_masks)

            iou, dice = calculate_iou_dice(mask_preds, query_image_masks)
            ap = calculate_ap(mask_preds, query_image_masks)

            iou = sum(iou) / (len(iou))
            dice = sum(dice) / (len(dice))
            ap = sum(ap) / (len(ap))

            self.log("val/overall_loss", overall_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/dice", dice, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/ap", ap, on_step=False, on_epoch=True, prog_bar=True)

            if batch_index % 5 == 0:
                self.validation_step_outputs.append({
                    'images': [batch['prompt_image'][:2], batch['query_image'][:2]],
                    'preds': [mask_preds[:2], mask_preds[:2]],
                    'masks': [prompt_image_masks[:2], query_image_masks[:2]]
                })

            return mask_preds, mask_preds

    def on_train_epoch_start(self):
        cur_epoch = self.trainer.current_epoch

        # if cur_epoch == 0:
        #     self.trainer.train_dataloader.loaders.dataset.image_augmentations.set_augmentation_level(3)
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = self.lr

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        for output in self.validation_step_outputs[:4]:
            prompt_image, query_image = output['images'][0], output['images'][1]
            prompt_preds, query_preds = output['preds'][0], output['preds'][1]
            prompt_masks, query_masks = output['masks'][0], output['masks'][1]

            vis = visualize(prompt_image, query_image, query_preds, query_masks)

            self.logger.experiment.add_image(
                f'val/predictions',
                vis,
                global_step=self.current_epoch
            )

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_index, dataloader_index=0):
        with torch.no_grad():
            mask_preds = self(batch)
            prompt_image_masks, query_image_masks = batch["prompt_image_target_masks"], batch[
                "query_image_target_masks"]

            return [mask_preds, prompt_image_masks], [mask_preds, query_image_masks]

    def test_epoch_end(self, multiple_test_set_outputs):
        """
        Test set evaluation function.
        """
        if len(self.test_set_names) == 1:
            multiple_test_set_outputs = [multiple_test_set_outputs]
        # iterate over all the test sets
        for test_set_name, test_set_batch_outputs in zip(self.test_set_names, multiple_test_set_outputs):
            predicted_masks = []
            target_masks = []
            # iterate over all the batches for the current test set
            for test_set_outputs in test_set_batch_outputs:
                left, right = test_set_outputs
                left_preds, left_targets = left
                right_preds, right_targets = right
                predicted_masks.extend([left_preds, right_preds])
                target_masks.extend([left_targets, right_targets])

            # # compute metrics
            # ap_map_precision_recall = eval_detection_voc(
            #     predicted_bboxes, target_bboxes, iou_thresh=0.5
            # )

            metrics = {'iou': [], 'dice': [], 'ap': []}
            for idx in range(len(predicted_masks)):
                iou, dice = calculate_iou_dice(predicted_masks[idx], target_masks[idx])
                ap = calculate_ap(predicted_masks[idx], target_masks[idx])
                metrics['iou'].extend(iou)
                metrics['dice'].extend(dice)
                metrics['ap'].extend(ap)

            metrics['iou'] = sum(metrics['iou']) / len(metrics['iou'])
            metrics['dice'] = sum(metrics['dice']) / len(metrics['dice'])
            metrics['ap'] = sum(metrics['ap']) / len(metrics['ap'])

            L.log(
                "INFO",
                f"{test_set_name} IOU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, AP: {metrics['ap']:.4f}",
            )

    def configure_optimizers(self):
        optimizer_params = [
            {"params": [parameter for parameter in self.parameters() if parameter.requires_grad]}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_params, lr=self.lr, weight_decay=self.weight_decay)
        return self.optimizer

    def forward(self, batch):
        return self.model(batch)


from PIL import Image, ImageDraw


def convert_segmentation_to_mask(image_width, image_height, segmentations):
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    for segmentation in segmentations:
        if len(segmentation) % 2 != 0 or len(segmentation) < 6:
            continue

        points = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
        draw.polygon(points, outline=255, fill=255)
    mask_array = np.array(mask, dtype=np.uint8)
    return mask_array


IMAGE_SIZE = (256, 256)


def marshal_getitem_data(data, split):
    """
    The data field above is returned by the individual datasets.
    This function marshals that data into the format expected by this
    model/method.
    """
    if split in ["train", "val", "test"]:
        (
            data["image1"],
            target_region_and_annotations,
        ) = utils.geometry.resize_image_and_annotations(
            data["image1"],
            output_shape_as_hw=IMAGE_SIZE,
            annotations=data["image1_target_annotations"],
        )
        data["image1_target_annotations"] = target_region_and_annotations
        (
            data["image2"],
            target_region_and_annotations,
        ) = utils.geometry.resize_image_and_annotations(
            data["image2"],
            output_shape_as_hw=IMAGE_SIZE,
            annotations=data["image2_target_annotations"],
        )
        data["image2_target_annotations"] = target_region_and_annotations

    assert data["image1"].shape == data["image2"].shape
    image1_width, image1_height = data['image1'].shape[-2:]
    image2_width, image2_height = data['image2'].shape[-2:]
    image1_segmentations = [x['segmentation'][0] for x in data['image1_target_annotations']]
    image2_segmentations = [x['segmentation'][0] for x in data['image2_target_annotations']]
    image1_target_masks = torch.Tensor(convert_segmentation_to_mask(image1_width, image1_height, image1_segmentations))
    image2_target_masks = torch.Tensor(convert_segmentation_to_mask(image2_width, image2_height, image2_segmentations))

    image1_target_masks = image1_target_masks.unsqueeze(dim=0) / 255.0
    image2_target_masks = image2_target_masks.unsqueeze(dim=0) / 255.0

    return {
        "prompt_image": data["image1"],
        "query_image": data["image2"],
        "prompt_image_target_masks": image1_target_masks,
        "query_image_target_masks": image2_target_masks,
        # "target_bbox_labels": torch.zeros(len(image1_target_masks)).long(),
        "query_metadata": {
            "pad_shape": data["image1"].shape[-2:],
            "border": np.array([0, 0, 0, 0]),
            "batch_input_shape": data["image1"].shape[-2:],
        },
    }


def dataloader_collate_fn(batch):
    """
    Defines the collate function for the dataloader specific to this
    method/model.
    """
    keys = batch[0].keys()
    collated_dictionary = {}
    for key in keys:
        collated_dictionary[key] = []
        for batch_item in batch:
            collated_dictionary[key].append(batch_item[key])
        if key in [
            "prompt_image_target_maskss",
            "query_image_target_maskss",
            "query_metadata",
            "target_bbox_labels",
        ]:
            continue
        collated_dictionary[key] = ImageList.from_tensors(
            collated_dictionary[key], size_divisibility=32
        ).tensor
    return collated_dictionary


class CallbackManager(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        datamodule = DataModule(args)
        datamodule.setup()
        self.test_set_names = datamodule.test_dataset_names

    @rank_zero_only
    def on_fit_start(self, trainer, model):
        if self.args.no_logging:
            return
        # trainer.logger.experiment.config.update(self.args, allow_val_change=True)

    @rank_zero_only
    def on_validation_batch_end(
            self, trainer, model, predicted_masks, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:
            self.val_batch = batch
            left_predicted, right_predicted = predicted_masks
            self.val_set_predicted_masks = [left_predicted, right_predicted]
            self.val_set_target_masks = [batch["prompt_image_target_masks"], batch["query_image_target_masks"]]

    @rank_zero_only
    def on_validation_end(self, trainer, model):
        self.log_qualitative_predictions(
            self.val_batch,
            self.val_set_predicted_masks,
            self.val_set_target_masks,
            "val",
            trainer,
        )

    @rank_zero_only
    def on_test_start(self, trainer, model):
        self.test_batches = [[] for _ in range(len(self.test_set_names))]
        self.test_set_predicted_masks = [[] for _ in range(len(self.test_set_names))]
        self.test_set_target_masks = [[] for _ in range(len(self.test_set_names))]

    @rank_zero_only
    def on_test_batch_end(
            self, trainer, model, predicted_and_target_masks, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:
            self.test_batches[dataloader_idx] = batch
            left, right = predicted_and_target_masks
            left_preds, left_targets = left
            right_preds, right_targets = right

            self.test_set_predicted_masks[dataloader_idx] = [left_preds, right_preds]
            self.test_set_target_masks[dataloader_idx] = [left_targets, right_targets]

    @rank_zero_only
    def on_test_end(self, trainer, model):
        for test_set_index, test_set_name in enumerate(self.test_set_names):
            self.log_qualitative_predictions(
                self.test_batches[test_set_index],
                self.test_set_predicted_masks[test_set_index],
                self.test_set_target_masks[test_set_index],
                f"test_{test_set_name}",
                trainer,
            )

    def log_qualitative_predictions(
            self,
            batch,
            predicted_masks,
            target_masks,
            batch_name,
            trainer,
    ):
        """
        Logs the predicted masks for a single val/test batch for qualitative analysis.
        """
        b, c, h, w = batch["prompt_image"].shape

        left_iou, left_dice = calculate_iou_dice(predicted_masks[0], target_masks[0])
        right_iou, right_dice = calculate_iou_dice(predicted_masks[1], target_masks[1])

        mean_left_iou, mean_left_dice = sum(left_iou) / len(left_iou), sum(left_dice) / len(left_dice)
        mean_right_iou, mean_right_dice = sum(right_iou) / len(right_iou), sum(right_dice) / len(right_dice)

        vis = visualize(batch["prompt_image"], batch["query_image"], predicted_masks[0], target_masks[1])

        if trainer.logger is not None:
            trainer.logger.experiment.add_image(
                'validation/comparison',
                vis,
                global_step=trainer.global_step,
            )

        # for i in range(b):
        #     prompt_image, left_preds, left_target = batch["prompt_image"][i], target_masks[0][i], predicted_masks[0][i]
        #     query_image, right_preds, right_target = batch["query_image"][i], target_masks[1][i], predicted_masks[1][i]
        #
        #     left_preds, left_target = left_preds.squeeze(), left_target.squeeze()
        #     right_preds, right_target = right_preds.squeeze(), right_target.squeeze()
        #
        #     vis = visualize_segmentation_comparison(prompt_image,left_preds,left_target,
        #                                                  query_image,right_preds,right_target)
        #
        #     if trainer.logger is not None:
        #         trainer.logger.experiment.add_image(
        #             'validation/comparison',
        #             vis,
        #             global_step=trainer.global_step,
        #         )

        L.log("INFO", f"Finished computing qualitative predictions for {batch_name}." +
              f"Left IOU: {mean_left_iou:.4f},Left Dice: {mean_left_dice:.4f}" +
              f"Right IOU: {mean_right_iou:.4f},Right Dice: {mean_right_dice:.4f}")
