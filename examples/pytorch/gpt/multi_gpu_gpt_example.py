# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

import argparse
import configparser
import os
import sys

import torch
from torch.nn.utils.rnn import pad_sequence

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))
from examples.pytorch.gpt.utils import comm
from examples.pytorch.gpt.utils import gpt_decoder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=1,
                        help='input sequence length to generate.')
    parser.add_argument('--output_len', type=int, default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--head_num', type=int, default=16,
                        help='head number')
    parser.add_argument('--size_per_head', type=int, default=64,
                        help='size per head')
    parser.add_argument('--vocab_size', type=int, default=50304,
                        help='vocab size')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='../models/megatron-models/c-model/345m/1-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--start_id', type=int, default=50256,
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=50256,
                        help='end token id.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='max batch size.')
    parser.add_argument('--min_length', type=int, default=0,
                        help='A minimum number of tokens to generate')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--inference_data_type', '--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument(
        '--weights_data_type',
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help='Data type of FT checkpoint weights',
    )
    
    parser.add_argument('--start_batch_size', type=int, default=8)
    parser.add_argument('--end_batch_size', type=int, default=8)
    parser.add_argument('--batch_size_hop', type=int, default=8)
    parser.add_argument('--profile_iters', type=int, default=8)


    args = parser.parse_args()

    ckpt_config = configparser.ConfigParser()
    ckpt_config_path = os.path.join(args.ckpt_path, 'config.ini')
    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)
    if 'gpt' in ckpt_config.keys():
        for args_key, config_key, func in [
            ('max_seq_len', 'max_pos_seq_len', ckpt_config.getint),
            ('weights_data_type', 'weight_data_type', ckpt_config.get),
        ]:
            if config_key in ckpt_config['gpt'].keys():
                args.__dict__[args_key] = func('gpt', config_key)
        for key in ['head_num', 'size_per_head', 'tensor_para_size']:
            if key in args.__dict__:
                args.__dict__[key] = ckpt_config.getint('gpt', key)


    layer_num = 1
    output_len = args.output_len
    head_num = args.head_num
    size_per_head = args.size_per_head
    vocab_size = args.vocab_size
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    start_id = args.start_id
    end_id = args.end_id
    max_seq_len = args.max_seq_len

    torch.manual_seed(0)

    comm.initialize_model_parallel(args.tensor_para_size, args.pipeline_para_size)

    # Prepare model.
    gpt = gpt_decoder.Gpt(
        num_heads=head_num,
        size_per_head=size_per_head,
        num_layers=layer_num,
        vocab_size=vocab_size,
        start_id=start_id,
        end_id=end_id,
        tensor_para_size=tensor_para_size,
        pipeline_para_size=pipeline_para_size,
        lib_path = args.lib_path,
        max_seq_len=max_seq_len,
        int8_mode=0,
        weights_data_type=args.weights_data_type)
    gpt.load(args.ckpt_path, args.inference_data_type)

    for batch_size in range (args.end_batch_size, args.start_batch_size - 1, -args.batch_size_hop):
        start_ids = [torch.IntTensor([end_id for _ in range(args.input_len)])] * batch_size
        start_lengths = [len(ids) for ids in start_ids]
        start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
        start_lengths = torch.IntTensor(start_lengths)
        
        gpt.generate(input_token_ids=start_ids,
                    input_lengths=start_lengths,
                    gen_length=output_len,
                    local_batch_size=batch_size,
                    profile_iters=args.profile_iters)
        
        del start_ids
        del start_lengths
 

if __name__ == '__main__':
    main()
