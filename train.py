import os
import sys # Keep for now, though direct sys.exit might be less common with Hydra
import glob # Keep if used for other things, not for config

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import torch

# from sgmse.model import StochasticRegenerationModel
# from FLOWTSE.datasets.librimix import SpecsDataModuleTSE

from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import random

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig # Added for accessing Hydra's output dir

OmegaConf.register_new_resolver("sgmse", lambda x: x) # Added for Hydra to recognize sgmse package

import logging

log = logging.getLogger(__name__)

def seed_everything(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)

# The main logic will be wrapped in a function decorated by @hydra.main
@hydra.main(config_path="config", config_name="example_config", version_base=None)
def main(cfg: DictConfig) -> None:
	seed_everything(cfg.seed)

	if cfg.logstdout:
		# Basic stdout redirection, consider Hydra's logging for more robust solution
		sys.stdout = open('stdout.log', 'w')
		sys.stderr = open('stderr.log', 'w')

	# print(OmegaConf.to_yaml(cfg)) # Log the effective configuration

	model = hydra.utils.instantiate(cfg.model) # Recursive instantiation

	hydra_output_dir = HydraConfig.get().runtime.output_dir
	logger = TensorBoardLogger(save_dir=hydra_output_dir, name="", version="tb")

	### Callbacks ###
	callbacks = []
	# callbacks.append(pl.callbacks.RichProgressBar())
	callbacks.append(pl.callbacks.TQDMProgressBar())

	# checkpoint_dir = os.path.join(hydra_output_dir, "checkpoints") # Define checkpoint directory
	# save_top_k == -1  <-- saves all models
	callbacks.append(ModelCheckpoint(
		dirpath=os.path.join(hydra_output_dir, "best_valid_loss"), 
		save_last=False, 
		save_top_k=1, 
		monitor="valid_loss",
		filename="epoch{epoch:03d}_validloss_{valid_loss:.2f}",
		auto_insert_metric_name=False,
	))

	callbacks.append(ModelCheckpoint(
		dirpath=os.path.join(hydra_output_dir, "best_sisdr"), 
		save_last=True, 
		save_top_k=10, 
		monitor="ValidationSISDR",
		mode="max",
		filename="epoch{epoch:03d}_sisdr_{ValidationSISDR:.2f}",
		auto_insert_metric_name=False,
	))

	callbacks.append(ModelCheckpoint(
		dirpath=os.path.join(hydra_output_dir, "best_pesq"), 
		save_last=False, 
		save_top_k=1, 
		monitor="ValidationPESQ",
		mode="max",
		filename="epoch{epoch:03d}_pesq_{ValidationPESQ:.2f}",
		auto_insert_metric_name=False,
	))

	callbacks.append(ModelCheckpoint(
		dirpath=os.path.join(hydra_output_dir, "best_estoi"), 
		save_last=False, 
		save_top_k=1, 
		monitor="ValidationESTOI",
		mode="max",
		filename="epoch{epoch:03d}_estoi_{ValidationESTOI:.2f}",
		auto_insert_metric_name=False,
	))

	# callbacks.append(ModelCheckpoint(
	# 	dirpath=checkpoint_dir, 
	# 	save_top_k=1, monitor="ValidationPESQ", mode="max", filename='epoch{epoch:03d}_pesq_{ValidationPESQ:.2f}',
	# 	auto_insert_metric_name=False
	# ))

	# Save checkpoints every N epochs for evaluation
	callbacks.append(ModelCheckpoint(
		dirpath=os.path.join(hydra_output_dir, "epoch_checkpoints"), 
		save_last=False, 
		save_top_k=-1,  # Save all checkpoints
		every_n_epochs=5,  # Save every 5 epochs
		filename="epoch{epoch:03d}",
		auto_insert_metric_name=False,
	))
	##################
	
	trainer = hydra.utils.instantiate(
		cfg.trainer_args, callbacks=callbacks, logger=logger, 
	)

	# pretrained_ckpt_path: loads only the weights
	# resume_from_checkpoint: loads the entire model state including optimizer, scheduler, etc.
	pretrained_ckpt_path = getattr(cfg, "pretrained_ckpt_path", None)

	if pretrained_ckpt_path is not None:
		pretrained_ckpt_path = hydra.utils.to_absolute_path(pretrained_ckpt_path)
		log.info(f"Loading pretrained weights from: {pretrained_ckpt_path}")
		checkpoint = torch.load(pretrained_ckpt_path, map_location=model.device, weights_only=False)
		model.load_state_dict(checkpoint['state_dict'])
		# Load EMA state if available
		if 'ema' in checkpoint:
			model.ema.load_state_dict(checkpoint['ema'])
			log.info("EMA state loaded from pretrained checkpoint")
		else:
			log.warning("No EMA state found in pretrained checkpoint")
		ckpt_path = None
	else:
		# Check for a resume checkpoint in the config
		ckpt_path = getattr(cfg, "resume_from_checkpoint", None)
		if ckpt_path is not None:
			ckpt_path = hydra.utils.to_absolute_path(ckpt_path)

	# Training and testing
	if cfg.get('train', True):  # Default to True if not specified
		log.info("Starting training...")
		if ckpt_path is not None:
			log.info(f"Resuming training from checkpoint: {ckpt_path}")
			trainer.fit(model, ckpt_path=ckpt_path)
		else:
			trainer.fit(model)

	if cfg.get('test', False):  # Default to False if not specified
		try:
			log.info("Starting testing...")
			trainer.test(model, ckpt_path="best")
		except Exception as e:
			log.error(f"Test with best model failed: {e}")
			log.info("Testing with current model instead...")
			trainer.test(model)

if __name__ == '__main__':
	main()

# example command line to run this script:

# python train.py --config-name=example_config
# python train.py --config-name=128freq_small
# python train.py --config-name=256freq_small_usef
# python train.py --config-name=256freq_large_usef

# python /home/minyeong.jeong/flowSS/FLOWTSE_fromscratch/train.py --config-name=your_config_file_name
# (Ensure 'your_config_file_name.yaml' is in the '/home/minyeong.jeong/flowSS/FLOWTSE_fromscratch/config/' directory)
# To override parameters:
# python /home/minyeong.jeong/flowSS/FLOWTSE_fromscratch/train.py --config-name=your_config_file_name trainer.max_epochs=50 model.lr=0.001

