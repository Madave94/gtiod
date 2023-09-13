import argparse
import os
import mmcv
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)
from mmdet.apis.train import auto_scale_lr
from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)
from gtiod.utils import add_root_dir_to_dataset_config

def parse_args():
    parser = argparse.ArgumentParser(description="Train MASK R-CNN on the TexBiG or VinDr-CXR dataset.")
    parser.add_argument("config", help="Path to config file.")
    parser.add_argument("dataset_root", help="Path to dataset root")

    parser.add_argument("--gpus", default=None, type=int, nargs="+",
                        help="Set gpus different to the ones mentioned in config. This overwrites the config file GPU.")
    parser.add_argument("--distributed", action="store_true", help="set this flag active for distributed training")
    parser.set_defaults(distributed=False)
    parser.add_argument("--no_validate", action="store_true", help="set this flag active to deactivate validation runs")
    parser.set_defaults(no_validate=False)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)

    args = parser.parse_args()

    return args

def main():
    """
        This is mainly the config from mmdet/apis/train.py which is just here to allow adaptability
    """
    args = parse_args()

    # load and check config compatibility
    config = mmcv.Config.fromfile(args.config)
    config = compat_cfg(config)
    config = add_root_dir_to_dataset_config(config, args.dataset_root)
    if args.gpus is not None:
        config.gpu_ids = args.gpus

    # parsing of optional arguments
    distributed = args.distributed
    validate = not args.no_validate
    meta = args.meta
    timestamp = args.timestamp

    # create folder
    mmcv.mkdir_or_exist(os.path.abspath(config.work_dir))

    # initialize logger
    logger = get_root_logger(log_file=os.path.join(config.work_dir, "train.log"), log_level=config.log_level)

    # prepare dataset and data loaders
    dataset = [build_dataset(config.data.train)]
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # runner init
    runner_type = 'EpochBasedRunner' if 'runner' not in config else config.runner[
        'type']

    model = build_detector(config.model)

    # add classes to checkpoint storing
    if config.checkpoint_config is not None:
        config.checkpoint_config.meta = dict(CLASSES = dataset[0].CLASSES)
        model.CLASSES = dataset[0].CLASSES

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(config.gpu_ids),
        dist=distributed,
        seed=config.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **config.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = config.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            config.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, config.device, device_ids=config.gpu_ids)

    # build optimizer
    auto_scale_lr(config, distributed, logger)
    optimizer = build_optimizer(model, config.optimizer)

    runner = build_runner(
        config.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=config.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = config.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **config.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in config.optimizer_config:
        optimizer_config = OptimizerHook(**config.optimizer_config)
    else:
        optimizer_config = config.optimizer_config

    # register hooks
    runner.register_training_hooks(
        config.lr_config,
        optimizer_config,
        config.checkpoint_config,
        config.log_config,
        config.get('momentum_config', None),
        custom_hooks_config=config.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **config.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            config.data.val.pipeline = replace_ImageToTensor(
                config.data.val.pipeline)
        val_dataset = build_dataset(config.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = config.get('evaluation', {})
        eval_cfg['by_epoch'] = config.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if config.resume_from is None and config.get('auto_resume'):
        resume_from = find_latest_checkpoint(config.work_dir)
    if resume_from is not None:
        config.resume_from = resume_from

    if config.resume_from:
        runner.resume(config.resume_from)
    elif config.load_from:
        runner.load_checkpoint(config.load_from)
    runner.run(data_loaders, config.workflow)

if __name__ == '__main__':
    main()