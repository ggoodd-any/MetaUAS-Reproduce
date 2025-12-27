import warnings
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from easydict import EasyDict
from loguru import logger as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from utils.general import get_easy_dict_from_yaml_file

from data.datamodule import DataModule
from metauas_wrapper import MetaUASWrapper

warnings.filterwarnings("ignore")


@rank_zero_only
def print_args(configs):
    L.log("INFO", configs)


def train(configs, model, logger, datamodule, callbacks=None):
    torch.autograd.set_detect_anomaly(True)

    L.log("INFO", f"Training model.")
    trainer = pl.Trainer.from_argparse_args(
        configs,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        benchmark=False,
        profiler="simple",
        # accumulate_grad_batches=16, # if batchsize=8
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer, trainer.checkpoint_callback.best_model_path


def test(configs, model, logger, datamodule, checkpoint_path, callbacks=None):
    L.log("INFO", f"Testing model.")
    tester = pl.Trainer.from_argparse_args(
        configs, logger=logger, callbacks=callbacks, benchmark=True
    )
    tester.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)


def get_logging_callback_manager(args):
    from metauas_wrapper import CallbackManager
    return CallbackManager(args)


if __name__ == "__main__":

    # For RTX 5090
    # torch.set_float32_matmul_precision('highest')

    parser = ArgumentParser()
    parser.add_argument("--no_logging", action="store_true", default=False)
    parser.add_argument("--test_from_checkpoint", type=str, default="")
    parser.add_argument("--quick_prototype", action="store_true", default=False)
    parser.add_argument("--load_weights_from", type=str, default=None)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--experiment_name", type=str, default=None)
    args, _ = parser.parse_known_args()
    parser = MetaUASWrapper.add_model_specific_args(parser)

    # parse configs from cmd line and config file into an EasyDict
    parser = DataModule.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args = EasyDict(vars(args))
    configs = get_easy_dict_from_yaml_file(args.config_file)

    # copy cmd line configs into config file configs, overwriting any duplicates
    for key in args.keys():
        if args[key] is not None:
            configs[key] = args[key]
        elif key not in configs.keys():
            configs[key] = args[key]

    if configs.quick_prototype:
        configs.limit_train_batches = 2
        configs.limit_val_batches = 2
        configs.limit_test_batches = 2
        configs.max_epochs = 1

    print_args(configs)

    pl.seed_everything(1, workers=True)

    datamodule = DataModule(configs)
    model = MetaUASWrapper(configs)

    logger = None
    callbacks = [get_logging_callback_manager(configs)]
    if not configs.no_logging:
        logger = TensorBoardLogger(
            save_dir="./tb_logs",
            name=args.experiment_name,
            version='v2',
            default_hp_metric=False,
            log_graph=True

        )

        callbacks.append(ModelCheckpoint(monitor="val/iou", mode="max", save_last=True))

    trainer = None
    if configs.test_from_checkpoint == "":
        # train the model and store the path to the best model (as per the validation set)
        # Note: multi-GPU training is supported.
        trainer, test_checkpoint_path = train(configs, model, logger, datamodule, callbacks)
        # test the best model exactly once on a single GPU
        torch.distributed.destroy_process_group()
    else:
        # test the given model checkpoint
        test_checkpoint_path = configs.test_from_checkpoint

    configs.gpus = 1
    if trainer is None or trainer.global_rank == 0:
        test(
            configs,
            model,
            logger,
            datamodule,
            test_checkpoint_path if test_checkpoint_path != "" else None,
            callbacks,
        )
