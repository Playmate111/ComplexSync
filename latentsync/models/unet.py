# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/models/unet.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin

from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from .resnet import InflatedConv3d, InflatedGroupNorm

from ..utils.util import zero_rank_log
from .utils import zero_module


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _strip_peft_prefix(key: str) -> str:
    """Remove the PEFT wrapper prefix from a checkpoint key (at most once)."""
    for prefix in ("base_model.model.", "base_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def normalize_and_merge_peft_state_dict(state_dict, lora_rank=64, lora_alpha=64):
    """Convert a plain or PEFT-wrapped checkpoint into plain UNet weights.

    PEFT training checkpoints store the frozen weights under `base_layer.*`
    and the low-rank delta under `lora_A.default.weight` / `lora_B.default.weight`.
    Those deltas must be merged into the base weights, otherwise the released
    weights cannot be loaded into `UNet3DConditionModel`.

    The input dictionary is never modified.
    """
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected the checkpoint state dict to be a mapping, got {type(state_dict)!r}")

    plain_state_dict = {}
    lora_tensors = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        if ".lora_A." in key or ".lora_B." in key:
            letter = "A" if ".lora_A." in key else "B"
            module = key.split(f".lora_{letter}.", 1)[0]
            lora_tensors.setdefault(module, {})[letter] = value
            continue

        # PEFT stores wrapped base parameters under `base_layer`.
        normalized_key = _strip_peft_prefix(key.replace(".base_layer.", "."))
        plain_state_dict[normalized_key] = value

    for module, tensors in lora_tensors.items():
        if "A" not in tensors or "B" not in tensors:
            raise RuntimeError(
                f"Incomplete LoRA pair for {module!r}: found {sorted(tensors)}; "
                "both lora_A and lora_B tensors are required."
            )

        lora_a, lora_b = tensors["A"], tensors["B"]
        target = _strip_peft_prefix(module + ".weight")
        if target not in plain_state_dict:
            raise RuntimeError(
                f"No base weight found for LoRA module {module!r} (looked for {target!r}). "
                "The checkpoint may use an unsupported PEFT layout."
            )

        base_weight = plain_state_dict[target]
        rank = lora_a.shape[0]
        if lora_rank is not None and rank != lora_rank:
            print(
                f"### warning: LoRA rank stored in the checkpoint is {rank}, "
                f"but the config specifies lora_rank={lora_rank}; using the checkpoint rank."
            )
        if lora_b.shape[0] != base_weight.shape[0]:
            raise RuntimeError(
                f"LoRA shape mismatch for {target!r}: lora_B has {lora_b.shape[0]} output channels "
                f"but the base weight has {base_weight.shape[0]}."
            )

        lora_b = lora_b.reshape(lora_b.shape[0], -1)
        if lora_b.shape[1] != rank:
            raise RuntimeError(
                f"LoRA shape mismatch for {target!r}: lora_B cannot be reduced to rank {rank} "
                f"(shape {tuple(tensors['B'].shape)})."
            )
        lora_a = lora_a.reshape(rank, -1)

        delta = (lora_b @ lora_a).reshape(base_weight.shape)
        plain_state_dict[target] = base_weight + delta.to(base_weight.dtype) * (lora_alpha / rank)

    if lora_tensors:
        print(f"### merged {len(lora_tensors)} LoRA modules (lora_alpha={lora_alpha})")

    if "conv_in.weight" not in plain_state_dict:
        sample_keys = list(state_dict.keys())[:8]
        looks_like_peft = any(
            key.startswith("base_model.model.") or ".lora_" in key for key in state_dict
        )
        raise RuntimeError(
            "Checkpoint normalization did not produce 'conv_in.weight'. "
            f"Detected {len(state_dict)} source keys (PEFT layout: {looks_like_peft}); "
            f"first keys: {sample_keys}. Supported formats are a plain UNet state dict "
            "and a PEFT checkpoint using 'base_model.model.*', 'base_layer.*', "
            "'lora_A.default.*' and 'lora_B.default.*' names."
        )

    return plain_state_dict


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        use_inflated_groupnorm=False,
        # Additional
        use_motion_module=False,
        motion_module_resolutions=(1, 2, 4, 8),
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type=None,
        motion_module_kwargs={},
        add_audio_layer=False,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        self.use_motion_module = use_motion_module
        self.add_audio_layer = add_audio_layer
        self.in_channels = in_channels

        self.conv_in = zero_module(InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2**i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions)
                and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                add_audio_layer=add_audio_layer,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                add_audio_layer=add_audio_layer,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                add_audio_layer=add_audio_layer,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
        self.conv_act = nn.SiLU()

        self.conv_out = zero_module(InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1))

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def set_gradient_checkpointing(self, value=False):
        for name, module in self.named_modules():
            if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
                print(f">>> Setting `gradient_checkpointing={value}` for '{name}'")
                module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor = None,
        motion_switchs: torch.Tensor = None,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # support controlnet
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre-process
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    motion_switchs=motion_switchs,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    motion_switchs=motion_switchs
                )

            down_block_res_samples += res_samples

        # support controlnet
        down_block_res_samples = list(down_block_res_samples)
        if down_block_additional_residuals is not None:
            for i, down_block_additional_residual in enumerate(down_block_additional_residuals):
                if down_block_additional_residual.dim() == 4:  # boardcast
                    down_block_additional_residual = down_block_additional_residual.unsqueeze(2)
                down_block_res_samples[i] = down_block_res_samples[i] + down_block_additional_residual

        # mid
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            motion_switchs=motion_switchs,
            attention_mask=attention_mask
        )

        # support controlnet
        if mid_block_additional_residual is not None:
            if mid_block_additional_residual.dim() == 4:  # boardcast
                mid_block_additional_residual = mid_block_additional_residual.unsqueeze(2)
            sample = sample + mid_block_additional_residual

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    motion_switchs=motion_switchs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    encoder_hidden_states=encoder_hidden_states,
                    motion_switchs=motion_switchs
                )

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

    @classmethod
    def from_pretrained(cls, model_config: dict, ckpt_path: str, device="cpu"):
        model_config = dict(model_config)
        # These are checkpoint-loader settings, not UNet architecture settings,
        # so they are consumed before `from_config`.
        lora_rank = model_config.pop("lora_rank", 64)
        lora_alpha = model_config.pop("lora_alpha", 64)

        unet = cls.from_config(model_config).to(device)
        resume_global_step = 0

        if ckpt_path != "":
            zero_rank_log(logger, f"Load from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            if isinstance(ckpt, dict) and "global_step" in ckpt:
                zero_rank_log(logger, f"resume from global_step: {ckpt['global_step']}")
                resume_global_step = ckpt["global_step"]

            # Released checkpoints use both layouts: a plain UNet state dict and
            # a PEFT training checkpoint wrapped in a "state_dict" entry.
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            state_dict = normalize_and_merge_peft_state_dict(
                state_dict, lora_rank=lora_rank, lora_alpha=lora_alpha
            )

            allowed_missing = set()
            conv_in_weight = state_dict.get("conv_in.weight")
            if conv_in_weight is not None and conv_in_weight.shape[1] != model_config["in_channels"]:
                print(
                    f"### warning: checkpoint conv_in.weight has {conv_in_weight.shape[1]} input channels, "
                    f"but the config expects {model_config['in_channels']}; "
                    "conv_in will be left at its initial value."
                )
                state_dict.pop("conv_in.weight", None)
                state_dict.pop("conv_in.bias", None)
                allowed_missing = {"conv_in.weight", "conv_in.bias"}

            missing, unexpected = nn.Module.load_state_dict(unet, state_dict, strict=False)
            print(f"### missing keys: {len(missing)}; \n### unexpected keys: {len(unexpected)};")

            unexplained_missing = [key for key in missing if key not in allowed_missing]
            if unexplained_missing:
                raise RuntimeError(
                    f"Failed to load {len(unexplained_missing)} required keys from {ckpt_path}. "
                    f"First missing keys: {unexplained_missing[:8]}."
                )
            if unexpected:
                print(f"### warning: ignoring unexpected checkpoint keys, e.g. {list(unexpected)[:8]}")

            del ckpt
            torch.cuda.empty_cache()

        return unet, resume_global_step
