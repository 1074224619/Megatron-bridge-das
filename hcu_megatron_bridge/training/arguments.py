# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Megatron arguments."""

import os
import argparse
from typing import Union

from megatron.training.arguments import add_megatron_arguments
from megatron.training.utils import warn_rank_0
from megatron.core.msc_utils import MultiStorageClientFeature


def remove_original_params(parser, param_names: Union[list, str]):
    if isinstance(param_names, str):
        param_names = [param_names]

    for action in parser._actions:
        if action.dest in param_names:
            parser._actions.remove(action)
            for option_string in action.option_strings:
                if option_string in parser._option_string_actions:
                    del parser._option_string_actions[option_string]


def process_adaptor_args(parser):
    from hcu_megatron_bridge.features_manager import ADAPTOR_FEATURES

    for feature in ADAPTOR_FEATURES:
        feature.register_args(parser)

    return parser


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    parser = add_megatron_arguments(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # add adaptor args
    parser = process_adaptor_args(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    args._is_global_batch_size_explicitly_specified = args.global_batch_size is not None

    # Experimental yaml
    if args.yaml_cfg is not None:
        from megatron.training.yaml_arguments import load_yaml

        args = load_yaml(args.yaml_cfg)


    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    # Args to enable MSC (opt-in: disabled by default)
    if args.disable_msc_deprecated:
        warn_rank_0(
            '--disable-msc is deprecated and will be removed in a future release. '
            'MSC is now disabled by default; pass --enable-msc to opt in.'
        )
        # Preserve legacy semantics: --disable-msc forces MSC off, even if
        # --enable-msc was also passed.
        args.enable_msc = False

    if args.enable_msc:
        MultiStorageClientFeature.enable()
        if not MultiStorageClientFeature.is_enabled():
            raise RuntimeError(
                "--enable-msc was passed but the multistorageclient package is not "
                "installed. Install it with `pip install multi-storage-client`."
            )
        warn_rank_0('The MSC feature is enabled.')
    else:
        MultiStorageClientFeature.disable()
        assert MultiStorageClientFeature.is_enabled() is False

    return args


_ADAPTOR_ARGS = None

def get_adaptor_args():
    global _ADAPTOR_ARGS
    if _ADAPTOR_ARGS is None:
        parser = argparse.ArgumentParser(description='Adaptor Arguments', allow_abbrev=False)
        _ADAPTOR_ARGS, _ = process_adaptor_args(parser).parse_known_args()

    return _ADAPTOR_ARGS


def destroy_adaptor_args():
    global _ADAPTOR_ARGS
    _ADAPTOR_ARGS = None