# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Tongzhou Wang
# Licensed under the MIT License.

import contextlib
from typing import Any, Dict, Generator, List, Optional

import torch
import torch.nn as nn
from torch.distributed.utils import _replace_by_prefix

from .flat_param import FlatParameter, FlatParamHandle, HandleConfig

FLAT_PARAM = "flat_param"
FPW_MODULE = "_fpw_module"

__all__ = ["FlattenParamsWrapper"]


def _post_state_dict_hook(
    module: nn.Module, state_dict: Dict[str, Any], prefix: str, *args: Any
) -> Dict[str, Any]:
    """
    _post_state_dict_hook() is called after the state_dict() is executed
    and before returning the state_dict to the users.
    This API post-processes the keys of the state_dict to remove the
    FlattenParamsWrapper internal prefix.
    """
    # Move everything from FPW_MODULE up one level.
    _replace_by_prefix(state_dict, prefix + f"{FPW_MODULE}.", prefix)
    return state_dict


def _pre_load_state_dict_hook(
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> None:
    """
    _pre_load_state_dict_hook() is called before the _load_from_state_dict() is
    executed. This API pre-processes the keys of the state_dict to add the
    FlattenParamsWrapper internal prefix.
    """
    # Push everything down to FPW_MODULE level.
    _replace_by_prefix(state_dict, prefix, prefix + f"{FPW_MODULE}.")
    # The flat_param_* keys actually needs to move one level up.
    flat_param_key = prefix + f"{FPW_MODULE}.{FLAT_PARAM}"
    for k in list(state_dict.keys()):
        if k.startswith(flat_param_key):
            last_part = k.split(".")[-1]
            assert last_part.startswith(
                FLAT_PARAM
            ), f"Expected key to contain flat_param, but key name is {k}"
            _replace_by_prefix(state_dict, k, prefix + last_part)


class FlattenParamsWrapper(nn.Module):
    """
    This is a wrapper for flattening parameters in a ``nn.Module`` 's subtree
    into a single flattened parameter and is based on [1]. This is used for
    :class:`FullyShardedDataParallel` 's recursive wrapping.
    [1] https://github.com/SsnL/PyTorch-Reparam-Module

    Args:
        module (nn.Module): Module to wrap.
        params (List[nn.Parameter]): Parameters in ``module`` 's subtree to
            flatten into a single flattened parameter.
        device (torch.device): The compute and communication device for this
            wrapper's handle.
        config (HandleConfig): A config customizing this wrapper's handle based
            on FSDP's available features.

    Attributes:
        flat_param (Optional[FlatParameter]): The flattened parameter.
            ``flat_param`` is ``None`` either when (1) this wrapper manages no
            parameters or (2) the wrapped module's parameters are unflattened.
        _fpw_module (nn.Module): The wrapped module.
        _flat_param_handle (FlatParamHandle): A handle for the flattened
            parameter; only present if this wrapper manages parameters.
    """

    def __init__(
        self,
        module: nn.Module,
        params: List[nn.Parameter],
        device: torch.device,
        config: HandleConfig,
        use_orig_params: bool,
    ) -> None:
        super().__init__()
        self._fpw_module = module
        # Register hooks to clean parameter names for state dict (even if this
        # wrapper itself manages no parameters since it must clean names from
        # submodules)
        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(_pre_load_state_dict_hook)
        if len(params) == 0:
            return
        self._flat_param_handle = FlatParamHandle(
            params, module, device, config, use_orig_params
        )
        if not use_orig_params:
            self._register_flat_param()
        self._use_orig_params = use_orig_params
        assert getattr(self, FPW_MODULE) is self._fpw_module
        assert getattr(self, FLAT_PARAM) is self.flat_param

    @property
    def has_params(self) -> bool:
        """Returns whether this wrapper manages any parameters."""
        return hasattr(self, "_flat_param_handle")

    @property
    def flat_param(self) -> Optional[FlatParameter]:
        return self.handle.flat_param if self.has_params else None

    @property
    def handle(self) -> FlatParamHandle:
        assert hasattr(self, "_flat_param_handle"), (
            "Accessing the handle of a `FlattenParamsWrapper` that does not "
            "manage any parameters"
        )
        return self._flat_param_handle

    @property
    def module(self) -> Any:
        """Returns the wrapped module (like DDP)."""
        return self._fpw_module

    @contextlib.contextmanager
    def unflatten_as_params(self) -> Generator:
        """
        Assumes that the flattened parameter is unsharded. When in the context,
        de-registers the flattened parameter and unflattens the original
        parameters as ``nn.Parameter`` views into the flattened parameter.
        After the context, re-registers the flattened parameter and restores
        the original parameters as ``Tensor`` views into the flattened
        parameter.
        """
        if self.flat_param is None:
            yield
        else:
            self._deregister_flat_param()
            try:
                with self._flat_param_handle.unflatten_as_params():
                    yield
            finally:
                if not self.handle._use_orig_params:
                    self._register_flat_param()

    def _register_flat_param(self):
        """
        Registers the flattened parameter, making it visible to ``nn.Module``
        methods.

        We do not use :meth:`nn.Module.register_parameter` because we want
        ``flat_param`` to always be an attribute but dynamically change whether
        it is visible to ``nn.Module`` methods.
        """
        self._parameters["flat_param"] = self.flat_param

    def _deregister_flat_param(self):
        """
        De-registers the flattened parameter, hiding it from ``nn.Module``
        methods.

        We do not use ``del self.flat_param`` because we want ``flat_param`` to
        always be an attribute but dynamically change whether it is visible to
        ``nn.Module`` methods.
        """
        self._parameters.pop("flat_param", None)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes of this wrapper to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to `nn.Module`'s logic
        except AttributeError:
            return getattr(self.module, name)  # fall back to the wrapped module

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls to the wrapped module in case the wrapped
        module is an ``nn.Sequential``."""
        return self.module.__getitem__(key)

    def forward(self, *inputs: Any, **kwinputs: Any) -> Any:
        if self.flat_param is not None:
            # TODO (awgu): For `use_orig_params=True`, I have moved the
            # `_use_unsharded_views(False)` call to `handle.unshard()`, namely
            # `_use_unsharded_flat_param()`. When we retire FPW, we should
            # consolidate `use_orig_params=False` to do the same.
            if not self._use_orig_params:
                self._flat_param_handle._use_unsharded_views(as_params=False)
        return self.module(*inputs, **kwinputs)
