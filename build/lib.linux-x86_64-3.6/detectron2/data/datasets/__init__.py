# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .cityscapes import load_cityscapes_instances
from .soba import load_soba_json, load_sem_seg
from .lvis import load_lvis_json, register_lvis_instances, get_lvis_instances_meta
from .register_soba import register_soba_instances
from . import builtin  # ensure the builtin datasets are registered


__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
