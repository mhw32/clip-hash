import os, wandb, getpass
import random, torch, numpy
from src.systems import systems
from src.utils.setup import process_config
import pytorch_lightning as pl

torch.backends.cudnn.benchmark = True

SYSTEM = {
    'PretrainSystem': systems.PretrainSystem,
    'TransferSystem': systems.TransferSystem,
}


def run(config_path, gpu_device=None):
    if gpu_device == 'cpu' or not gpu_device:
        gpu_device = None

    config = process_config(config_path)

    if gpu_device: 
        config.gpu_device = int(gpu_device)

    seed_everything(config.seed, use_cuda=config.cuda)
    SystemClass = SYSTEM[config.system]
    system = SystemClass(config)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=-1,
        period=1,
    )

    wandb.init(
        project='clip_hash', 
        entity=getpass.getuser(), 
        name=config.exp_name, 
        config=config, 
        sync_tensorboard=True,
    )

    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=gpu_device,
        max_epochs=config.num_epochs,
        min_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        resume_from_checkpoint=config.continue_from_checkpoint,
        gradient_clip_val=0.5,
    )
    trainer.fit(system)


def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--gpu-device', default=None)
    args = parser.parse_args()

    gpu_device = str(args.gpu_device) if args.gpu_device else None
    run(args.config, gpu_device=args.gpu_device)
