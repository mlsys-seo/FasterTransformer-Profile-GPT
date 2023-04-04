# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import abstractmethod
from pathlib import Path
from typing import List, Literal, Optional, Union
import os

import numpy as np
import torch

from . import comm
from . import profiler
from .gpt import GptInitModelParameters


PathLike = Union[str, Path]


def to_numpy_dtype(maybe_str_dtype: Union[str, np.dtype]):
    assert isinstance(maybe_str_dtype, (str, np.dtype))
    if isinstance(maybe_str_dtype, str):
        try:
            dtype = {
                'fp16': np.float16,
                'float16': np.float16,
                'fp32': np.float32,
                'float32': np.float32,
            }[maybe_str_dtype]
        except KeyError:
            raise ValueError(f'Cannot convert to numpy data type, got {maybe_str_dtype}')
    else:
        dtype = maybe_str_dtype
    return dtype


def to_torch_dtype(maybe_str_dtype: Union[str, torch.dtype]):

    if isinstance(maybe_str_dtype, torch.dtype):
        dtype = maybe_str_dtype
    else:
        try:
            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[maybe_str_dtype]
        except KeyError:
            raise ValueError(f"Cannot convert to torch data type, got {maybe_str_dtype}")
    return dtype


def load_weight_from_bin(checkpoint_path: PathLike,
                         shape: List[int],
                         weight_dtype: Union[str, np.dtype]):
    """ Load a weight from a bin file.

    # Args.
        checkpoint_path: str or Path, a checkpoint file path of an FT's layer weight.
        shape: list of int, the shape of weight tensor.
        weight_dtype: str or np.dtype, the data type of the stored weight.
    """
    weight_dtype = to_numpy_dtype(weight_dtype)
    return torch.from_numpy(np.fromfile(checkpoint_path, dtype=weight_dtype))

LayernormType = Literal['pre_layernorm', 'post_layernorm']


class GptLayerWeights:

    def __init__(self,
                 num_heads: int,
                 size_per_head: int,
                 inter_size: int,
                 num_layers: int,
                 tensor_para_size: int = 1,
                 pipeline_para_size: int = 1,
                 has_adapters: bool = False,
                 adapter_inter_size: int = 0,
                 int8_mode: int = 0):

        assert num_heads % tensor_para_size == 0, \
            f'num_heads ({num_heads}) is not multiple of tensor para size ({tensor_para_size})'

        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.hidden_units = num_heads * size_per_head
        self.num_layers = num_layers

        self.tensor_para_size = tensor_para_size
        self.tensor_para_rank = comm.get_tensor_para_rank()
        self.pipeline_para_size = pipeline_para_size
        self.pipeline_para_rank = comm.get_pipeline_para_rank()

        self.has_adapters = has_adapters
        self.adapter_inter_size = adapter_inter_size

        self.local_num_layers = num_layers // pipeline_para_size
        self.local_num_heads = num_heads // tensor_para_size
        self.local_hidden_units = self.local_num_heads * size_per_head
        self.local_inter_size = inter_size // tensor_para_size
        self.local_adapter_inter_size = self.adapter_inter_size // tensor_para_size

        self.weight_transpose_calibrate_quantize = None
        assert int8_mode in [0, 1], "Invalid int8 mode for GPT. Must be 0 or 1"
        self.int8_mode = int8_mode
        if self.int8_mode == 1:
            quant = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix
            self.weight_transpose_calibrate_quantize = lambda x : quant(x, torch.int8)

        self.weights = None
        self.int8_weights = None
        self.int8_scales = None

        self.expected_weight_shapes = list()

        # pylint:disable=line-too-long
        # Transformer blocks
        self.expected_weight_shapes.extend([(self.hidden_units,)] * self.local_num_layers) # input layernorm weight
        self.expected_weight_shapes.extend([(self.hidden_units,)] * self.local_num_layers) # input layernorm bias
        self.expected_weight_shapes.extend([(self.hidden_units, self.local_hidden_units * 3)] * self.local_num_layers) # attention qkv weight
        self.expected_weight_shapes.extend([(self.local_hidden_units * 3,)] * self.local_num_layers) # attention qkv bias
        self.expected_weight_shapes.extend([(self.local_hidden_units, self.hidden_units)] * self.local_num_layers) # attention dense weight
        self.expected_weight_shapes.extend([(self.hidden_units,)] * self.local_num_layers) # attention dense bias
        self.expected_weight_shapes.extend([(self.hidden_units,)] * self.local_num_layers) # post attention layernorm weight
        self.expected_weight_shapes.extend([(self.hidden_units,)] * self.local_num_layers) # post attention layernorm bias
        self.expected_weight_shapes.extend([(self.hidden_units, self.local_inter_size)] * self.local_num_layers) # ffn_kernel1
        self.expected_weight_shapes.extend([(self.local_inter_size,)] * self.local_num_layers)  # ffn_bias1
        self.expected_weight_shapes.extend([(self.local_inter_size, self.hidden_units)] * self.local_num_layers) # ffn_kernel2
        self.expected_weight_shapes.extend([(self.hidden_units,)] * self.local_num_layers)  # ffn_bias2

        # Adapters
        if self.has_adapters:
            self.expected_weight_shapes.extend([(self.hidden_units, self.local_adapter_inter_size)] * self.local_num_layers) # adaptor1_kernel1
            self.expected_weight_shapes.extend([(self.local_adapter_inter_size,)] * self.local_num_layers)  # adaptor1_bias1
            self.expected_weight_shapes.extend([(self.local_adapter_inter_size, self.hidden_units)] * self.local_num_layers) # adaptor1_kernel2
            self.expected_weight_shapes.extend([(self.hidden_units,)] * self.local_num_layers)  # adaptor1_bias2
            self.expected_weight_shapes.extend([(self.hidden_units, self.local_adapter_inter_size)] * self.local_num_layers) # adaptor2_kernel1
            self.expected_weight_shapes.extend([(self.local_adapter_inter_size,)] * self.local_num_layers)  # adaptor2_bias1
            self.expected_weight_shapes.extend([(self.local_adapter_inter_size, self.hidden_units)] * self.local_num_layers) # adaptor2_kernel2
            self.expected_weight_shapes.extend([(self.hidden_units,)] * self.local_num_layers)  # adaptor2_bias2
        # pylint:enable=line-too-long


    @classmethod
    def from_config(cls, config: GptInitModelParameters):
        return cls(num_heads=config.head_num,
                   size_per_head=config.size_per_head,
                   inter_size=4 * config.head_num * config.size_per_head,
                   num_layers=config.layer_num,
                   tensor_para_size=config.tensor_para_size,
                   pipeline_para_size=config.pipeline_para_size,
                   has_adapters=config.has_adapters,
                   adapter_inter_size=config.adapter_inter_size,
                   int8_mode=config.int8_mode)

    @property
    def dtype(self):
        return self.weights[0].dtype

    @property
    def device(self):
        return self.weights[0].device

    def _map(self, func):
        for i in range(len(self.weights)):
            if isinstance(self.weights[i], list):
                for j in range(len(self.weights[i])):
                    self.weights[i][j] = func(self.weights[i][j])
            else:
                self.weights[i] = func(self.weights[i])

    def _map_int8(self, func):
        for i in range(len(self.int8_weights)):
            if isinstance(self.int8_weights[i], list):
                for j in range(len(self.int8_weights[i])):
                    self.int8_weights[i][j] = func(self.int8_weights[i][j])

            else:
                self.int8_weights[i] = func(self.int8_weights[i])
        for i in range(len(self.int8_scales)):
            if isinstance(self.int8_scales[i], list):
                for j in range(len(self.int8_scales[i])):
                    self.int8_scales[i][j] = func(self.int8_scales[i][j])
            else:
                self.int8_scales[i] = func(self.int8_scales[i])

    def float(self):
        if self.dtype == torch.float32:
            return
        self._map(lambda x: x.float())

    def half(self):
        if self.dtype == torch.float16:
            return
        self._map(lambda x: x.half())
        if self.int8_mode == 1:
            self._map_int8(lambda w: w.half())

    def bfloat16(self):
        if self.dtype == torch.bfloat16:
            return
        self._map(lambda x: x.bfloat16())
        if self.int8_mode == 1:
            self._map_int8(lambda w: w.bfloat16())

    def cuda(self, device=None):
        self._map(lambda x: x.cuda(device))
        if self.int8_mode == 1:
            self._map_int8(lambda x: x.cuda(device))

    def to(self, device=None):
        self._map(lambda x: x.to(device))
        if self.int8_mode == 1:
            self._map_int8(lambda x: x.to(device))

    def is_valid_pp_group(self, layer, pp_rank):
        return layer // self.layers_per_device == pp_rank

    def load(self,
             checkpoint_path: PathLike,
             compute_dtype: torch.dtype,
             weight_dtype: Optional[Union[str, np.dtype]] = None,
             device: Optional[Union[int, str, torch.device]] = None):
        """ Load checkpoint weights.

        # Args.
            checkpoint_path: str or Path,
                a checkpoint directory where FT checkpoint files locate.
            weight_dtype: str or np.dtype, the data type of stored weights.
        """

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Could not find checkpoint {str(checkpoint_path)}")

        weight_dtype = to_numpy_dtype(weight_dtype)
        print(f'Load weights from {str(checkpoint_path)} (data type: {weight_dtype}')

        self.weights      = list()
        self.int8_weights = list()
        self.int8_scales  = list()
        torch.cuda.empty_cache()

        def _load_from_file(fname):
            quant_sub_names = ["attention.query_key_value.weight", "attention.dense.weight", \
                               "dense_h_to_4h.weight", "dense_4h_to_h.weight"]
            _weight = torch.from_numpy(np.fromfile(checkpoint_path / fname, dtype=weight_dtype))
            _weight = _weight.to(compute_dtype)
            weight_index   = len(self.weights)
            expected_shape = self.expected_weight_shapes[weight_index]

            try:
                if _weight.nelement() > 0:
                    _weight = _weight.reshape(expected_shape)
            except:
                raise ValueError(
                    f"num_heads, size_per_head, vocab_size, and max_seq_len must be the same "
                    f"as the ones during training (weight: {fname} expected shape: {expected_shape}, "
                    f"got shape: {_weight.shape}).")

            should_quantize = any(sub_name in fname for sub_name in quant_sub_names)
            if self.int8_mode != 0 and should_quantize:
                calibrate = self.weight_transpose_calibrate_quantize
                int8_weight, int8_scales = calibrate(_weight)

                #int8 weights should appear in same order as FP weights.
                # Move to device and add to the int8 list.
                dummy_weight = torch.empty(0, dtype=compute_dtype)
                if device is not None:
                    int8_weight = int8_weight.to(device)
                    int8_scales = int8_scales.to(device)
                    dummy_weight = dummy_weight.to(device)

                self.int8_weights.append(int8_weight)
                self.int8_scales.append(int8_scales)
                self.weights.append(dummy_weight)
            else:
                if device is not None:
                    _weight = _weight.to(device)
                self.weights.append(_weight)

        # Load
        # pylint:disable=line-too-long
        layer_offset = self.local_num_layers * self.pipeline_para_rank
        [_load_from_file(f'model.layers.0.input_layernorm.weight.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.input_layernorm.bias.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.attention.query_key_value.weight.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.attention.query_key_value.bias.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.attention.dense.weight.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.attention.dense.bias.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.post_attention_layernorm.weight.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.post_attention_layernorm.bias.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.mlp.dense_h_to_4h.weight.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.mlp.dense_h_to_4h.bias.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.mlp.dense_4h_to_h.weight.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
        [_load_from_file(f'model.layers.0.mlp.dense_4h_to_h.bias.bin') for i in range(self.local_num_layers)]

        if self.has_adapters:
            [_load_from_file(f'model.layers.0.after_attention_adapter.dense_h_to_4h.weight.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
            [_load_from_file(f'model.layers.0.after_attention_adapter.dense_h_to_4h.bias.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
            [_load_from_file(f'model.layers.0.after_attention_adapter.dense_4h_to_h.weight.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
            [_load_from_file(f'model.layers.0.after_attention_adapter.dense_4h_to_h.bias.bin') for i in range(self.local_num_layers)]
            [_load_from_file(f'model.layers.0.after_ffn_adapter.dense_h_to_4h.weight.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
            [_load_from_file(f'model.layers.0.after_ffn_adapter.dense_h_to_4h.bias.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
            [_load_from_file(f'model.layers.0.after_ffn_adapter.dense_4h_to_h.weight.{self.tensor_para_rank}.bin') for i in range(self.local_num_layers)]
            [_load_from_file(f'model.layers.0.after_ffn_adapter.dense_4h_to_h.bias.bin') for i in range(self.local_num_layers)]

        assert len(self.weights) == len(self.expected_weight_shapes), "Incorrect number of weights loaded"


class FtModuleBase:

    def __init__(self):
        self.weight = None

    @classmethod
    @abstractmethod
    def from_config(cls, config: GptInitModelParameters, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _initialize_model(self, force_init=False):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_weight(self, weight: GptLayerWeights):
        old_weight_dtype = self.weight.dtype if self.weight is not None else None
        self.weight = weight
        if old_weight_dtype is None or old_weight_dtype != self.weight.dtype:
            self._initialize_model(force_init=True)

    @property
    def dtype(self):
        assert self.weight is not None
        return self.weight.dtype

    @property
    def device(self):
        assert self.weight is not None
        return self.weight.device

    def cuda(self, device=None):
        assert torch.cuda.is_available()
        self.weight.cuda(device)
        return self

    def to(self, device=None):
        self.weight.to(device)
        return self

    def float(self):
        self.weight.float()
        self._initialize_model(force_init=True)
        return self

    def half(self):
        self.weight.half()
        self._initialize_model(force_init=True)
        return self

    def bfloat16(self):
        self.weight.bfloat16()
        self._initialize_model(force_init=True)
        return self


class GptContextDecoder(FtModuleBase):

    def __init__(self,
                 num_heads: int,
                 size_per_head: int,
                 inter_size: int,
                 num_layers: int,
                 tensor_para_size: int = 1,
                 pipeline_para_size: int = 1,
                 remove_padding: bool = True,
                 shared_contexts_ratio: float = 1.0,
                 layernorm_eps: float = 1e-6,
                 layernorm_type: LayernormType = 'pre_layernorm',
                 activation_type: str = 'gelu',
                 has_adapters: bool = False,
                 adapter_inter_size: int = 0,
                 int8_mode: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.hidden_size = self.num_heads * self.size_per_head
        self.inter_size = inter_size
        self.num_layers = num_layers

        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size

        self.remove_padding = remove_padding
        self.shared_contexts_ratio = shared_contexts_ratio

        self.layernorm_eps = layernorm_eps
        self.layernorm_type = layernorm_type
        self.activation_type = activation_type
        self.has_adapters = has_adapters
        self.adapter_inter_size = adapter_inter_size

        assert int8_mode in [0, 1]
        self.int8_mode = int8_mode

        self.ft_op = None
        self.weight = None

    def __repr__(self):
        args_dict = dict(
            num_heads=self.num_heads,
            size_per_head=self.size_per_head,
            hidden_size=self.hidden_size,
            inter_size=self.inter_size,
            num_layers=self.num_layers,
            tensor_para_size=self.tensor_para_size,
            pipeline_para_size=self.pipeline_para_size,
            remove_padding=self.remove_padding,
            shared_contexts_ratio=self.shared_contexts_ratio,
            layernorm_eps=self.layernorm_eps,
            layernorm_type=self.layernorm_type,
            activation_type=self.activation_type,
            has_adapters=self.has_adapters,
            adapter_inter_size=self.adapter_inter_size,
            int8_mode=self.int8_mode,
        )
        args_str = ',\n    '.join([f'{k}: {v}' for k, v in args_dict.items()])
        return f'{self.__class__.__name__}[\n{    args_str}\n]'

    @classmethod
    def from_config(cls, config: GptInitModelParameters, **kwargs):
        return cls(num_heads=config.head_num,
                   size_per_head=config.size_per_head,
                   inter_size=4 * config.head_num * config.size_per_head,
                   num_layers=config.layer_num,
                   tensor_para_size=config.tensor_para_size,
                   pipeline_para_size=config.pipeline_para_size,
                   remove_padding=kwargs.get('remove_padding', True),
                   shared_contexts_ratio=kwargs.get('shared_contexts_ratio', 1.0),
                   layernorm_eps=config.layernorm_eps,
                   layernorm_type=config.layernorm_type,
                   activation_type=config.activation_type,
                   has_adapters=config.has_adapters,
                   adapter_inter_size=config.adapter_inter_size,
                   int8_mode=config.int8_mode)

    def _initialize_model(self, force_init=False):
        if self.weight is None:
            self.weight = GptLayerWeights(
                num_heads=self.num_heads,
                size_per_head=self.size_per_head,
                inter_size=self.inter_size,
                num_layers=self.num_layers,
                tensor_para_size=self.tensor_para_size,
                pipeline_para_size=self.pipeline_para_size,
                has_adapters=self.has_adapters,
                adapter_inter_size=self.adapter_inter_size,
                int8_mode=self.int8_mode)
        if not force_init and self.ft_op is not None:
            return
        if self.ft_op is not None:
            del self.ft_op

        self.ft_op = torch.classes.FasterTransformer.ParallelGptContextDecoderOp(
            self.num_heads,
            self.size_per_head,
            self.inter_size,
            self.num_layers,
            self.tensor_para_size,
            self.pipeline_para_size,
            self.layernorm_eps,
            self.layernorm_type,
            self.activation_type,
            self.has_adapters,
            self.adapter_inter_size,
            self.int8_mode,
            self.weight.weights,
            self.weight.int8_weights,
            self.weight.int8_scales,
            self.remove_padding)

    def forward(self,
                input_embeds: torch.Tensor,
                attention_mask: torch.Tensor,
                input_lengths: torch.IntTensor,
                memory_length: Optional[int] = None,
                compact_index: Optional[torch.IntTensor] = None,
                batch_to_compact_index: Optional[torch.IntTensor] = None,
                linear_bias_slopes: Optional[torch.Tensor] = None,
                profile_iters: int = 2):
        """

        # Args.
            input_embeds: Tensor, (batch * beam, max_input_length, hidden_dim),
                input hidden states.
            attention_mask: Tensor, (batch * beam, max_input_length, max_input_length),
                input attention mask.
            input_lengths: (batch * beam,), input sequence lengths.
            memory_length: int, the length of memory to keep key/cache values.
            compact_index: IntTensor, (compact_batch_size,)
                The index of input sequences of a compact batch. If None, the FT op
                doesn't apply the shared context feature and as result the inference
                time may increase.
            batch_to_compact_index: IntTensor, (batch * beam,)
                The index map from the original input batch to the compact batch.
                This must be provided if compact_index is not None.
            linear_bias_slopes: (num_heads,)
                The slope per head of linear attention bias - ALiBi. If None, a base
                self attention will be performed.
        # Returns
            hidden_states: Tensor, (batch * beam, max_input_length, hidden_dim),
                decoder outputs.
            key_cache: Tensor, (num_layers, batch * beam, local_num_heads, size_per_head / x, memory_length, x),
                key cache of attention of inputs.
                x = 16 / sizeof(T), memory_length = max_input_length or max_input_length + gen_length
            value_cache: Tensor, (num_layers, batch * beam, local_num_heads, memory_length, hidden_dim)
                value cache of attention
            last_token_hidden_states: Tensor, (batch * beam, hidden_dim)
                hidden states of the last input token.
        """
        self._initialize_model()
        # outputs: output hidden states
        decoder_ouptut, key_cache, value_cache, last_token_hidden_states = self.ft_op.forward(
            input_embeds,
            attention_mask,
            input_lengths,
            memory_length,
            compact_index,
            batch_to_compact_index,
            linear_bias_slopes,
            profile_iters)
        return decoder_ouptut, key_cache, value_cache, last_token_hidden_states


class GptDecoder(FtModuleBase):

    def __init__(self,
                 num_heads: int,
                 size_per_head: int,
                 inter_size: int,
                 num_layers: int,
                 tensor_para_size: int = 1,
                 pipeline_para_size: int = 1,
                 layernorm_eps: float = 1e-6,
                 layernorm_type: LayernormType = 'pre_layernorm',
                 activation_type: str = 'gelu',
                 has_adapters: bool = False,
                 adapter_inter_size: int = 0,
                 int8_mode: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.hidden_size = self.num_heads * self.size_per_head
        self.inter_size = inter_size
        self.num_layers = num_layers

        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size

        self.layernorm_eps = layernorm_eps
        self.layernorm_type = layernorm_type
        self.activation_type = activation_type
        self.has_adapters = has_adapters
        self.adapter_inter_size = adapter_inter_size

        self.int8_mode = int8_mode

        self.ft_op = None
        self.weight = None

    def __repr__(self):
        args_dict = dict(
            num_heads=self.num_heads,
            size_per_head=self.size_per_head,
            hidden_size=self.hidden_size,
            inter_size=self.inter_size,
            num_layers=self.num_layers,
            tensor_para_size=self.tensor_para_size,
            pipeline_para_size=self.pipeline_para_size,
            layernorm_eps=self.layernorm_eps,
            layernorm_type=self.layernorm_type,
            activation_type=self.activation_type,
            has_adapters=self.has_adapters,
            adapter_inter_size=self.adapter_inter_size,
            int8_mode=self.int8_mode,
        )
        args_str = ',\n    '.join([f'{k}: {v}' for k, v in args_dict.items()])
        return f'{self.__class__.__name__}[\n    {args_str}\n]'

    @classmethod
    def from_config(cls, config: GptInitModelParameters, **kwargs):
        hidden_dim = config.head_num * config.size_per_head
        return cls(num_heads=config.head_num,
                   size_per_head=config.size_per_head,
                   inter_size=4 * hidden_dim,
                   num_layers=config.layer_num,
                   tensor_para_size=config.tensor_para_size,
                   pipeline_para_size=config.pipeline_para_size,
                   layernorm_eps=config.layernorm_eps,
                   layernorm_type=config.layernorm_type,
                   activation_type=config.activation_type,
                   has_adapters=config.has_adapters,
                   adapter_inter_size=config.adapter_inter_size,
                   int8_mode=config.int8_mode)

    def _initialize_model(self, force_init=False):
        if self.weight is None:
            self.weight = GptLayerWeights(
                num_heads=self.num_heads,
                size_per_head=self.size_per_head,
                inter_size=self.inter_size,
                num_layers=self.num_layers,
                tensor_para_size=self.tensor_para_size,
                pipeline_para_size=self.pipeline_para_size,
                has_adapters=self.has_adapters,
                adapter_inter_size=self.adapter_inter_size,
                int8_mode=self.int8_mode)
        if not force_init and self.ft_op is not None:
            return
        if self.ft_op is not None:
            del self.ft_op
        self.ft_op = torch.classes.FasterTransformer.ParallelGptDecoderOp(
            self.num_heads,
            self.size_per_head,
            self.inter_size,
            self.num_layers,
            self.tensor_para_size,
            self.pipeline_para_size,
            self.layernorm_eps,
            self.layernorm_type,
            self.activation_type,
            self.has_adapters,
            self.adapter_inter_size,
            self.weight.int8_mode,
            self.weight.weights,
            self.weight.int8_weights,
            self.weight.int8_scales)

    def forward(self,
                max_input_length: int,
                step: int,
                ite: int,
                input_embeds: torch.Tensor,
                sequence_lengths: torch.IntTensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                finished: torch.BoolTensor,
                total_padding_tokens: torch.IntTensor,
                masked_tokens: torch.BoolTensor,
                cache_indirection: Optional[torch.IntTensor] = None,
                linear_bias_slopes: Optional[torch.Tensor] = None,
                profile_iters: int = 2):
        """

        # Args.
            max_input_length: int, maximum input context length.
            step: int, the current step index.
            ite: int, local batch iteration.
            input_embeds: Tensor, (local_batch * beam, hidden_dim),
                input hidden state to decoder.
            sequence_lengths: IntTensor, (local_batch * beam,),
                the current sequence lengths.
            key_cache: Tensor, key cache buffer.
            value_cache: Tensor, value cache buffer.
            finished: BoolTensor, (local_batch * beam,),
                whether to finish sentence generation.
            total_padding_tokens IntTensor, (local_batch * beam,),
                the number of padded tokens.
            masked_tokens: BoolTensor, (local_batch * beam, memory_length),
                a mask tensor that indicates padded tokens.
            cache_indirection: IntTensor, (local_batch * beam,),
                cache of beam positions if needed if beam > 1.
            linear_bias_slopes Tensor, (num_heads,)
                slopes head of linear position bias (ALiBi) (optional).
        # Returns
            IntTensor, (batch * beam,) output token ids.
        """

        self._initialize_model()

        outputs = self.ft_op.forward(max_input_length,
                                     step,
                                     ite,
                                     input_embeds,
                                     sequence_lengths,
                                     finished,
                                     total_padding_tokens,
                                     masked_tokens,
                                     key_cache,
                                     value_cache,
                                     cache_indirection,
                                     linear_bias_slopes,
                                     profile_iters)
        return outputs[0]


class Gpt:

    def __init__(self,
                 num_heads: int,
                 size_per_head: int,
                 num_layers: int,
                 vocab_size: int,
                 start_id: int,
                 end_id: int,
                 lib_path: PathLike,
                 tensor_para_size: int = 1,
                 pipeline_para_size: int = 1,
                 remove_padding: bool = True,
                 shared_contexts_ratio: float = 1.0,
                 layernorm_eps: float = 1e-6,
                 layernorm_type: LayernormType = 'pre_layernorm',
                 activation_type: str = 'gelu',
                 has_positional_encoding: bool = True,
                 max_seq_len: int = 0,
                 has_pre_decoder_layernorm: bool = False,
                 has_post_decoder_layernorm: bool = True,
                 has_adapters: bool = False,
                 adapter_inter_size: int = 0,
                 int8_mode: int = 0,
                 inference_data_type: Optional[str] = None,
                 weights_data_type: str = 'fp32',
                 use_fp32_to_compute_logit: bool = False,
                 **kwargs):
        super().__init__()

        inference_data_type = inference_data_type or weights_data_type

        self.config = GptInitModelParameters(
            head_num=num_heads,
            size_per_head=size_per_head,
            layer_num=num_layers,
            max_seq_len=max_seq_len,
            tensor_para_size=tensor_para_size,
            vocab_size=vocab_size,
            start_id=start_id,
            end_id=end_id,
            pipeline_para_size=pipeline_para_size,
            data_type=inference_data_type,
            weights_data_type=weights_data_type,
            layernorm_eps=layernorm_eps,
            layernorm_type=layernorm_type,
            activation_type=activation_type,
            has_positional_encoding=has_positional_encoding,
            has_pre_decoder_layernorm=has_pre_decoder_layernorm,
            has_post_decoder_layernorm=has_post_decoder_layernorm,
            has_adapters=has_adapters,
            adapter_inter_size=adapter_inter_size,
            int8_mode=int8_mode,
            sparse=kwargs.get('sparse', False)
        )
        self.use_fp32_to_compute_logit = use_fp32_to_compute_logit

        self.weight = None
        self.shared_contexts_ratio = shared_contexts_ratio

        torch.classes.load_library(os.path.abspath(lib_path))

        # Embeddings to encode or decode tokens.
        hidden_dim = num_heads * size_per_head

        # Pad vocab size for FT.
        local_vocab_size = math.ceil(self.config.vocab_size / self.config.tensor_para_size)
        if self.config.data_type == 'fp16':
            local_vocab_size = math.ceil(local_vocab_size / 8) * 8
        self.vocab_size_padded = local_vocab_size * self.config.tensor_para_size
        self.vocab_size = self.config.vocab_size


        self._parameters = {}

        def register_param(name, p):
            self._parameters[name] = p
            setattr(self, name, p)

        register_param(
            'context_decoder',
            GptContextDecoder.from_config(
                self.config,
                remove_padding=remove_padding,
                shared_contexts_ratio=shared_contexts_ratio,
                **kwargs))
        register_param('decoder', GptDecoder.from_config(self.config, **kwargs))


    @classmethod
    def from_config(cls, config: GptInitModelParameters, **kwargs):
        return cls(
            num_heads=config.head_num,
            size_per_head=config.size_per_head,
            num_layers=config.layer_num,
            max_seq_len=config.max_seq_len,
            tensor_para_size=config.tensor_para_size,
            vocab_size=config.vocab_size,
            start_id=config.start_id,
            end_id=config.end_id,
            pipeline_para_size=config.pipeline_para_size,
            inference_data_type=config.data_type,
            weights_data_type=config.weights_data_type,
            layernorm_eps=config.layernorm_eps,
            layernorm_type=config.layernorm_type,
            activation_type=config.activation_type,
            has_positional_encoding=config.has_positional_encoding,
            has_pre_decoder_layernorm=config.has_pre_decoder_layernorm,
            has_post_decoder_layernorm=config.has_post_decoder_layernorm,
            has_adapters=config.has_adapters,
            adapter_inter_size=config.adapter_inter_size,
            int8_mode=config.int8_mode,
            **kwargs
        )

    def load(self,
             checkpoint_path: PathLike,
             inference_data_type: Optional[Union[str, torch.dtype]] = None,
             config: Optional[GptInitModelParameters] = None,
             device: Optional[Union[str, int, torch.device]] = None):

        checkpoint_path = Path(checkpoint_path)
        device = device or comm.get_device()
        config = config or self.config

        compute_dtype = to_torch_dtype(inference_data_type or self.dtype)

        self.weight = GptLayerWeights.from_config(config)
        self.weight.load(checkpoint_path, compute_dtype, config.weights_data_type, device)

        self.context_decoder.set_weight(self.weight)
        self.decoder.set_weight(self.weight)


        self.to(device)

    @property
    def dtype(self):
        assert self.weight is not None
        return self.weight.dtype

    @property
    def device(self):
        assert self.weight is not None
        return self.weight.device

    def cuda(self, device=None):
        assert torch.cuda.is_available()
        for name, param in self._parameters.items():
            setattr(self, name, param.cuda(device))
        return self

    def to(self, device=None):
        for name, param in self._parameters.items():
            setattr(self, name, param.to(device))
        return self

    def float(self):
        for name, param in self._parameters.items():
            setattr(self, name, param.float())
        return self

    def half(self):
        for name, param in self._parameters.items():
            setattr(self, name, param.half())
        return self

    def bfloat16(self):
        for name, param in self._parameters.items():
            setattr(self, name, param.bfloat16())
        return self

    def _mask_padded_vocab_weights(self, weight: torch.Tensor):
        assert self.vocab_size_padded >= self.vocab_size
        if self.vocab_size_padded > self.vocab_size:
            weight.data[self.vocab_size:, ...] = 0

    def generate_pad_mask(self, input_lengths, memory_length, init_step=0):
        """ Generate a pad mask tensor.

        # Args.
            input_lengths: (batch_size * beam_width,), input lengths
            memory_length: the length of key/value cache memory.
            init_step: int, initial step.
        # Return
            masked_tokens: BoolTensor, (batch_size * beam_width, memory_length),
                True if init_step + input_length[i] <= j < init_step + max_input_length,
                where i is a batch-beam index and j is a time step modulo by memory_length.
        """
        max_input_length = input_lengths.max()
        input_lengths = input_lengths.unsqueeze(1)
        shift = init_step % memory_length
        step_indices = torch.arange(
            init_step, init_step + memory_length, device=input_lengths.device)
        step_indices = step_indices.roll(shift).unsqueeze(0).tile(input_lengths.shape[0], 1)
        masked_tokens = torch.logical_and(
            step_indices >= input_lengths, step_indices < init_step + max_input_length)
        return masked_tokens

    def get_local_batch_size(self, batch_size):
        """ Get a local batch size by the same way that FT Gpt does. """
        local_batch_size = batch_size
        pp_size = self.decoder.pipeline_para_size
        if pp_size > 1:
            if local_batch_size % pp_size == 0:
                local_batch_size //= pp_size
            while local_batch_size > 1024 and local_batch_size % 2 == 0:
                local_batch_size //= 2
        return local_batch_size

    @torch.no_grad()
    def generate(self,
                 input_token_ids: torch.IntTensor,
                 input_lengths: torch.IntTensor,
                 gen_length: int,
                 local_batch_size: Optional[int] = None,
                 profile_iters: int = 2):
        """

        # Args.
            input_token_ids: IntTensor, (batch_size, max_input_length),
                input hidden state to decoder.
            input_lengths: IntTensor, (batch_size),
                the lengths of input context sequences.
            gen_length: int, the number of tokens to generate.
            local_batch_size: int, optional, a batch size of local iteration. (disabled)
            eos_token_id: int, eos token id.
            beam_width: int, number of beams for beam search.
                If 1, sampling decode will be used.
            top_k: IntTensor, (batch_size,) top-k sampling. The number of most probable
                tokens to keep for sampling per sentence in a batcch.
            top_p: FloatTensor, (batch_size,), top-p sampling. The cumulative probability
                of to filter the set of most probable tokens.
            top_p_decay: FloatTensor, (batch_size,)
                The decay of top-p value for top_p sampling.
            top_p_min: FloatTensor, (batch_size,)
                The minimum top p values in top-p decaying.
            top_p_reset_ids: IntTensor, (batch_size,)
                reset ids for resetting top_p values for top p sampling
            temperature: FloatTensor, (batch_size,),
                The temperature value for smoothing the logit distribution.
            repetition_penalty: FloatTensor, (batch_size,),
                The repetition penalty.
            presence_penalty: FloatTensor, (batch_size,),
                The presence penalty, which is exclusive with repetition_penalty.
                Only one of repetition and presence penalties is allowed.
            min_length: IntTensor, (batch_size,),
                Minimum length for each sentences. EOS is masked if length is below min.
            len_penalty: FloatTensor, (batch_size,)
                The exponent of the length penalty of beam scores.
            beam_search_diversity_rate: FloatTensor, (batch_size,),
                The diversity rate of beam search.
            stop_words_list: IntTensor, (batch_size, 2, stop_words_length)
                When FT generates words in this list, it will stop the generation. An extension of stop id.
            bad_words_list IntTensor, (batch_size, 2, bad_words_length)
                The words in the list will never be sampled.
            sequence_limit_lengths: IntTensor, (batch_size,), The maximum length of a generated sequence.
            memory_length: int, the length of cache memory. If None, it will be
                max_input_length + gen_length.
        # Returns
            IntTensor, (batch_size, beam_width, max_seq_length) output token ids.
        """
        assert self.weight is not None, 'Please call load() first to initialize weights.'

        input_token_ids = input_token_ids.type(torch.int32).to(self.device)
        input_lengths = input_lengths.type(torch.int32).to(self.device)

        batch_size = len(input_token_ids)
        max_input_length = input_token_ids.shape[-1]
        max_seq_length = max_input_length + gen_length

        assert local_batch_size is None or local_batch_size == batch_size
        local_batch_size = batch_size

        device = self.device

        pad_lengths = max_input_length - input_lengths
        # Since tril() doesn't support bf16 dtype, we create of bool type and then cast it to dtype.
        attention_mask = torch.ones(
            (max_input_length, max_input_length), dtype=torch.bool, device=device)\
            .tril().unsqueeze(0).tile(input_token_ids.shape[0], 1, 1).to(self.dtype)
        for b, input_length in enumerate(input_lengths):
            attention_mask[b, input_length:, ...] = 0
        masked_tokens = self.generate_pad_mask(input_lengths, max_seq_length)
        finished = torch.zeros_like(input_lengths).bool()
        sequence_lengths = (max_input_length - 1) * torch.ones_like(input_lengths)

        input_embeds = torch.empty(
            size=(batch_size, max_input_length, self.context_decoder.hidden_size),
            dtype=self.context_decoder.dtype,
            device=device)

        profiler.start('ft-context-decoder')
        _, k_cache, v_cache, last_token_hidden_states = self.context_decoder.forward(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            memory_length=max_seq_length,
            batch_to_compact_index=None,
            compact_index=None,
            profile_iters=profile_iters)
        profiler.stop('ft-context-decoder')

        step = max_seq_length - 1

        bbidx = range(
            0,
            min(local_batch_size, batch_size))

        input_embeds = torch.empty(
            size=(len(bbidx), self.decoder.hidden_size),
            dtype=self.decoder.dtype,
            device=device)

        profiler.start('ft-decoder')
        self.decoder.forward(
            max_input_length=max_input_length,
            step=step,
            ite=0,
            input_embeds=input_embeds,
            sequence_lengths=sequence_lengths[bbidx],
            key_cache=k_cache,
            value_cache=v_cache,
            finished=finished[bbidx],
            total_padding_tokens=pad_lengths[bbidx],
            cache_indirection=None,
            masked_tokens=masked_tokens[bbidx, ...],
            profile_iters=profile_iters)
        profiler.stop('ft-decoder')
