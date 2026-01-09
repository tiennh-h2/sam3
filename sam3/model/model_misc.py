# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Various utility models"""

import copy
import math
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager
from enum import auto, Enum
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import multi_head_attention_forward
from typing_extensions import override


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MultiheadAttentionWrapper(nn.MultiheadAttention):
    def forward(self, *args, **kwargs):
        kwargs["need_weights"] = False
        return super().forward(*args, **kwargs)


class ExportableMultiheadAttentionWrapper(MultiheadAttentionWrapper):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        r"""Compute attention outputs using query, key, and value embeddings.

            Supports optional parameters for padding, masks and attention weights.

        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and float masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
                and achieve the best performance for MHA.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
            is_causal: If specified, applies a causal mask as attention mask.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``attn_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """  # noqa: B950
        why_not_fast_path = ""
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None)
            and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
            key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = (
                "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
            )
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif nn.modules.activation._is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(nn.modules.activation._check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(
                nn.modules.activation._arg_requires_grad(x) for x in tensor_args
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class DotProductScoring(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, torch.nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp  # an optional MLP projection for prompt
        self.prompt_proj = torch.nn.Linear(d_model, d_proj)
        self.hs_proj = torch.nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
        is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
        # num_valid has shape (bs, 1)
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        # mean pool over all the valid tokens -- pooled_prompt has shape (bs, proj_dim)
        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        # hs has shape (num_layer, bs, num_query, d_model)
        # prompt has shape (seq, bs, d_model)
        # prompt_mask has shape (bs, seq), where 1 is valid and 0 is padding
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2

        # apply MLP on prompt if specified
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)

        # first, get the mean-pooled version of the prompt
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)

        # then, project pooled_prompt and hs to d_proj dimensions
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # (bs, d_proj)
        proj_hs = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # finally, get dot-product scores of shape (num_layer, bs, num_query, 1)
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale

        # clamp scores to a max value to avoid numerical issues in loss or matcher
        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",  # ["none"] only for now
        pos_enc_at_input_dec=True,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec

        # for two stage
        assert two_stage_type in ["none"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if (
                    "box_embed" not in n
                    and "query_embed" not in n
                    and "reference_points" not in n
                ):
                    nn.init.xavier_uniform_(p)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # whether to add the output as a residual connection to the input
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            torch._check((2*max(1, x.shape[2])) == (2*x.shape[2]))
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_clones_seq(module, N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_activation_module(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(pos_tensor, num_feats=256):
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class SAM3Output(list):
    """
    A class representing the output of a SAM3 model.
    It provides an iterable interface that supports different iteration modes, including iterating over all steps per stage,
    last step per stage, and flattened output.
    Attributes:
        output: The output of the SAM3 model, represented as a list of lists.
        iter_mode: The current iteration mode.
    Example:
        >>> output = [[1, 2], [3, 4], [5, 6]]
        >>> sam3_output = SAM3Output(output)
        >>> for step in sam3_output:
        ...     print(step)
        [1, 2]
        [3, 4]
        [5, 6]
        >>> with SAM3Output.iteration_mode(SAM3Output.IterMode.LAST_STEP_PER_STAGE) as sam3_last_step_out:
        ...     for step in sam3_last_step_out:
        ...         print(step)
        [2]
        [4]
        [6]
        >>> with SAM3Output.iteration_mode(SAM3Output.IterMode.FLATTENED) as sam3_flattened_out:
        ...     for step in sam3_flattened_out:
        ...         print(step)
        1
        2
        3
        4
        5
        6
    """

    class IterMode(Enum):
        # Defines the type of iterator over ouptuts.
        ALL_STEPS_PER_STAGE = auto()
        LAST_STEP_PER_STAGE = auto()
        FLATTENED = auto()  # Returns each interactivity step as if it is a separate stage (this is used in SAM3Image model)

    def __init__(
        self,
        output: List[List[Dict]] = None,
        iter_mode: IterMode = IterMode.ALL_STEPS_PER_STAGE,
        loss_stages: Optional[List[int]] = None,
    ):
        if output is not None:
            assert (
                isinstance(output, list)
                and len(output) > 0
                and isinstance(output[0], list)
            ), "Expected output to be a list of lists"
            self.output = output
        else:
            self.output = []
        assert isinstance(
            iter_mode, SAM3Output.IterMode
        ), f"iter_mode shoulf be of enum type 'SAM3Output.IterMode'. Got {type(iter_mode)}"

        self.iter_mode = iter_mode
        # We create a weak reference to self to be used in the lambda functions.
        # This is to avoid cyclic references and let SAM3Output be garabge collected.
        self_ref = weakref.ref(self)
        self._mode2iter = {
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE: lambda: iter(self_ref().output),
            SAM3Output.IterMode.LAST_STEP_PER_STAGE: lambda: (
                inner_list[-1] for inner_list in self_ref().output
            ),
            SAM3Output.IterMode.FLATTENED: lambda: (
                element for inner_list in self_ref().output for element in inner_list
            ),
        }
        self.loss_stages = loss_stages

    @override
    def __iter__(self) -> Iterator:
        return self._mode2iter[self.iter_mode]()

    def __getitem__(self, index):
        """
        Returns the item at the specified index.
        Args:
            index (int): The index of the item to return.
        Returns:
            list or element: The item at the specified index.
        """
        assert isinstance(index, int), f"index should be an integer. Got {type(index)}"
        if self.iter_mode == SAM3Output.IterMode.ALL_STEPS_PER_STAGE:
            return self.output[index]
        elif self.iter_mode == SAM3Output.IterMode.LAST_STEP_PER_STAGE:
            return self.output[index][-1]
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            if index == -1:
                return self.self.output[-1][-1]
            else:
                flattened_output = sum(self.output, [])
                return flattened_output[index]

    class _IterationMode(AbstractContextManager):
        """
        A context manager that temporarily changes the iteration mode of a SAM3Output object.
        This class is used internally by the SAM3Output.iteration_mode method.
        """

        def __init__(
            self, model_output: "SAM3Output", iter_mode: "SAM3Output.IterMode"
        ):
            self._model_output = model_output
            self._orig_iter_mode = model_output.iter_mode
            self._new_iter_mode = iter_mode

        @override
        def __enter__(self) -> "SAM3Output":
            self._model_output.iter_mode = self._new_iter_mode
            return self._model_output

        @override
        def __exit__(self, exc_type, exc_value, traceback):
            self._model_output.iter_mode = self._orig_iter_mode
            return super().__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def iteration_mode(
        model_output: "SAM3Output", iter_mode: IterMode
    ) -> _IterationMode:
        """
        Returns a context manager that allows you to temporarily change the iteration mode of the SAM3Output object.
        Args:
            model_output: The SAM3Output object.
            iter_mode: The new iteration mode.
        Returns:
            SAM3Output._IterationMode: A context manager that changes the iteration mode of the SAM3Output object.
        """
        return SAM3Output._IterationMode(model_output=model_output, iter_mode=iter_mode)

    def append(self, item: list):
        assert isinstance(
            item, list
        ), f"Only list items are supported. Got {type(item)}"
        self.output.append(item)

    def __repr__(self):
        return self.output.__repr__()

    def __len__(self):
        if self.iter_mode in [
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
            SAM3Output.IterMode.LAST_STEP_PER_STAGE,
        ]:
            return len(self.output)
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            flattened_output = sum(self.output, [])
            return len(flattened_output)
