# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
import numpy as np
import torch
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm

from .c2_model_loading import align_and_update_state_dicts


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        model_device = next(self.model.parameters()).device
        loaded = torch.load(filename, map_location=model_device)
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        self._convert_ndarray_to_tensor(checkpoint["model"])
        if checkpoint.get("matching_heuristics", False):
            model_device = next(self.model.parameters()).device
            for k in checkpoint["model"].keys():
                checkpoint["model"][k] = checkpoint["model"][k].clone().to(model_device)
            #print('model state dict', [(k, x.device) for k, x in model_state_dict.items()])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        # Move checkpoint to same device as model
        super()._load_model(checkpoint)

    def _convert_ndarray_to_tensor(self, state_dict: dict):
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        model_device = next(self.model.parameters()).device
        ones = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=model_device)
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(
                v, torch.Tensor
            ):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(
                        k, type(v)
                    )
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.tensor(v, device=model_device)
