#
# Copyright 2016 The BigDL Authors.
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
#

import torch
from torch import nn

from transformers import AutoTokenizer, PreTrainedTokenizerBase, LlamaConfig
from typing import Optional, Tuple, List, Type, Dict

from bigdl.llm.vllm.sequence import SequenceOutputs, SequenceGroupMetadata
from bigdl.llm.vllm.model_executor.layers.bigdl_sampler import BigDLSampler
from bigdl.llm.vllm.model_executor.models.bigdl_model import BigDLModelForCausalLM
from bigdl.llm.vllm.logger import init_logger
from bigdl.llm.vllm.model_executor.input_metadata import InputMetadata
import math
import time
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import torch.nn.functional as F


logger = init_logger(__name__)


def _pad_to_max(x: List[int], max_len: int, padding_id: int = 0) -> List[int]:
    return x + [padding_id] * (max_len - len(x))


def _padding_prompt_to_max(
    input_ids: List[List[int]], max_prompt_len: int, padding_id: int = 0
) -> List[List[int]]:
    return [
        prompt + [padding_id] * (max_prompt_len - len(prompt)) for prompt in input_ids
    ]


def _get_attention_mask_for_prompts(
    input_ids: List[List[int]], max_prompt_len: int
) -> List[List[int]]:
    attention_mask = [
        [1] * len(prompt) + [0] * (max_prompt_len - len(prompt)) for prompt in input_ids
    ]
    return attention_mask


class BigDLLlamaForCausalLM(BigDLModelForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        device: Optional[str] = None,
        max_model_len: Optional[int] = None,
        debug=True,
    ):
        super().__init__(config, device, max_model_len)
        self.config = config
        # TODO(gc): later change this to a switch?
        if True:
            from bigdl.llm.transformers import AutoModelForCausalLM
            from bigdl.llm import optimize_model

        # low_bit = 'sym_int4'
        if device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                config._name_or_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=True,
            )
            self.model = optimize_model(model)
            self.sampler = BigDLSampler(config.vocab_size, device)
        elif device == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
            except ImportError:
                print(
                    "Intel Extension for PyTorch is not installed, \
                       but is required for xpu inference."
                )

            low_bit = "sym_int4"
            model = AutoModelForCausalLM.from_pretrained(
                config._name_or_path,
                load_in_low_bit=low_bit,
                trust_remote_code=True,
                use_cache=True,
            )
            self.model = model.to("xpu")
            self.sampler = BigDLSampler(config.vocab_size, device).to("xpu")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = self.model.dtype
        self.last_seq_ids = []
        self.tmp_kv_cache = None
        self.pad_token_id = config.pad_token_id
        self.max_seq_limit = max_model_len
        self.debug = debug

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def forward_old(
        self,
        seq_group_meta_data_lists: List[SequenceGroupMetadata],
        kv_cache: Optional[List[List[Dict]]] = None,
        input_metadata: Optional[InputMetadata] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        kv_cache_size_0 = self.model.config.num_hidden_layers
        kv_cache_size_1 = 2
        seq_len = len(seq_group_meta_data_lists)

        # input_ids that will be passed to underlying models
        bigdl_input_ids = []
        bigdl_position_ids = []
        bigdl_attention_mask = []

        cur_seq_ids = []
        bigdl_sampling_params = {}
        max_context_len = 0
        all_decoding = True
        for seq_group_meta_data in seq_group_meta_data_lists:
            req_id = seq_group_meta_data.request_id
            all_decoding = all_decoding and (not seq_group_meta_data.is_prompt)
            seq_ids = list(seq_group_meta_data.seq_data.keys())
            seq_id = seq_ids[0]
            cur_seq_ids.append(seq_id)
            seq_data = seq_group_meta_data.seq_data[seq_id]

            cur_seq_input_ids = seq_data.get_token_ids()
            context_len = seq_data.get_len()
            if seq_group_meta_data.is_prompt:
                bigdl_input_ids.append(cur_seq_input_ids)
                max_context_len = max(max_context_len, context_len)
            else:
                bigdl_input_ids.append([cur_seq_input_ids[-1]])

            bigdl_sampling_params[seq_id] = seq_group_meta_data.sampling_params

        if all_decoding:
            bigdl_kv_cache = self.prepare_kv_cache(
                cur_seq_ids,
                seq_group_meta_data_lists,
                kv_cache,
                kv_cache_size_0,
                kv_cache_size_1,
            )
        else:
            bigdl_input_ids = [
                _pad_to_max(input_ids, max_context_len, self.pad_token_id)
                for input_ids in bigdl_input_ids
            ]

        if all_decoding:
            cur_seq_len = bigdl_kv_cache[0][0].size(2)
            for seq_group_meta_data in seq_group_meta_data_lists:
                seq_ids = list(seq_group_meta_data.seq_data.keys())
                seq_id = seq_ids[0]
                seq_data = seq_group_meta_data.seq_data[seq_id]
                cur_pos = seq_data.get_len()
                bigdl_position_ids.append([cur_pos - 1])
                cur_attention_mask = [0] * (cur_seq_len - cur_pos + 1) + [1] * (cur_pos)
                bigdl_attention_mask.append(cur_attention_mask)

        bigdl_input_ids = torch.tensor(bigdl_input_ids, device=self.device)
        if all_decoding:
            bigdl_position_ids = torch.tensor(bigdl_position_ids, device=self.device)
            bigdl_attention_mask = torch.tensor(
                bigdl_attention_mask, device=self.device
            )
            kwargs = {
                "input_ids": bigdl_input_ids,
                "position_ids": bigdl_position_ids,
                "attention_mask": bigdl_attention_mask,
                "past_key_values": bigdl_kv_cache,
                "use_cache": True,
                # "return_dict": True,
            }
        else:
            kwargs = {
                "input_ids": bigdl_input_ids,
                # "position_ids": bigdl_position_ids,
                "past_key_values": None,
                "use_cache": True,
                # "return_dict": True,
            }
        # pdb.set_trace()
        if self.device.type == "xpu":
            torch.xpu.empty_cache()
        st_timestamp = time.perf_counter()
        outputs = self.model.forward(**kwargs)
        # tmp = torch.xpu.memory_stats()
        # logger.info(f"0: {tmp['allocated_bytes.all.current']}")
        # self.last_seq_ids = cur_seq_ids[:]
        # self.last_kv_cache = outputs.past_key_values
        self._set_last_seq_ids(cur_seq_ids[:])
        self._set_last_kv_cache(outputs.past_key_values)

        logits = outputs.logits[:, -1, :]
        bigdl_output = self.sampler(logits, input_metadata, st_timestamp)
        # tmp = torch.xpu.memory_stats()
        # logger.info(f"before: {tmp['allocated_bytes.all.current']}")

        self.update_kv_cache(cur_seq_ids, kv_cache, kv_cache_size_0, kv_cache_size_1)

        # tmp = torch.xpu.memory_stats()
        # logger.info(f"after: {tmp['allocated_bytes.all.current']}")
        return bigdl_output

    def forward(
        self,
        seq_group_meta_data_lists: List[SequenceGroupMetadata],
        # kv_cache in the format [[dict() for _ in range(2)] for _ in range(32)]
        kv_cache: Optional[List[List[Dict]]] = None,
        input_metadata: Optional[InputMetadata] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # We should arrange inputs that will be passed to the bigdl-llm models
        # Specifically, we should arrange input_ids, position_ids(Optional), attention_mask(for prefill)
        # So, here it is:
        bigdl_input_ids = []
        bigdl_attention_mask = []
        processed_seq_ids = []

        # some parameters of model
        num_layers = self.model.config.num_hidden_layers

        # 0. Verify is_prompt or is_decoding
        is_prefill_stage = seq_group_meta_data_lists[0].is_prompt

        # 1. Assemble bigdl_input_ids
        for seq_group_meta_data in seq_group_meta_data_lists:
            # seq_group contains multiple sequences, we only process the first unfinished one
            processed_seq_id = list(seq_group_meta_data.seq_data.keys())[0]
            processed_seq_ids.append(processed_seq_id)
            processed_seq_data = seq_group_meta_data.seq_data[processed_seq_id]
            # We need to process differently for prefill/decoding
            if is_prefill_stage:
                # For prefill, all tokens are needed
                bigdl_input_ids.append(processed_seq_data.get_token_ids())
                self.debug_print("added one prompt to process...")
            else:
                # For decoding, only the last token is needed
                bigdl_input_ids.append([processed_seq_data.get_last_token_id()])

        self.debug_print("loop ends...")

        # For prompt we need to do padding so that they have the same length
        # TODO(gc): implement selective_batching for prefill
        # TODO(gc): test this logic with more than one prompt in parallel with variable length
        self.debug_print("#### Length of bigdl_input_ids:", len(bigdl_input_ids))
        if is_prefill_stage and len(bigdl_input_ids) > 1:
            self.debug_print("#### Doing padding")
            # padding bigdl_input_ids to max_prompt_len
            max_prompt_len = len(max(bigdl_input_ids, key=len))
            # attention_mask need to be calculated before padding
            bigdl_attention_mask = _get_attention_mask_for_prompts(
                bigdl_input_ids, max_prompt_len
            )
            bigdl_input_ids = _padding_prompt_to_max(bigdl_input_ids, max_prompt_len, 0)
        # 1. Assemble bigdl_input_ids end

        # 2. Assemble kv_cache for decoding stage?
        # TODO(gc): we may need to set bigdl_attention_mask correctly for decoding stage
        # TODO(gc): we need to pay attention to its devices.  Otherwise we may waste xpu memory
        # We need to extract KV_CACHE from our maintained kv_cache, and assemble it.

        # KV_CACHE should in the format (batch, num_heads, seq_len, embed_size_per_head)
        # Our maintained kv_cache in the format (num_heads, seq_len, embed_size_per_head)
        # However, differnet sequences may have different seq_len-> in different stage
        # We need to create a new kv_cache for handling:
        # a. iterate through the processed_seq_ids
        # b. for each of the processed_seq_id, get its kv_cache from the kv_cache, find the max_seq_len of those kv_cache
        # We only need to consider one layer and one key?
        # TODO(gc): This is correct if every layer has the same kv_cache shape, and key/value has the same kv_cache shape
        bigdl_kv_cache_list = [[] for _ in range(num_layers)]
        if not is_prefill_stage:
            max_kv_len = 0
            for processed_seq_id in processed_seq_ids:
                max_kv_len = max(max_kv_len, kv_cache[0][0][processed_seq_id].size(dim=1))
            # c. for each of the processed_seq_id, padding it to the max_seq_len, and concat
            for layer in range(num_layers):
                # Assemble KV_CACHE for this layer
                for kv in range(2):
                    # Assemble key_cache or value_cache
                    kv_list = []
                    for processed_seq_id in processed_seq_ids:
                        # Get its current kv_cache
                        processed_kv_cache = kv_cache[layer][kv][processed_seq_id]
                        # Added one more dimension to it
                        processed_kv_cache = processed_kv_cache.view([1] + list(processed_kv_cache.shape))
                        # Padding the tensor to max_length
                        if processed_kv_cache.size(dim=2) < max_kv_len:
                            pads = (0, 0, 0, max_kv_len - processed_kv_cache.size(dim=2), 0, 0, 0, 0)
                            processed_kv_cache = F.pad(processed_kv_cache, pads)
                        self.debug_print("padded kv_cache_size:", processed_kv_cache.shape)
                        kv_list.append(processed_kv_cache)
                    # key_cache = torch.cat(key_list, dim=0)
                    # value_cache = torch.cat(value_list, dim=0)
                    current_layer_kv_cache = torch.cat(kv_list, dim=0)
                    bigdl_kv_cache_list[layer].append(current_layer_kv_cache)
            # TODO(gc): for those paddings, how could we ensure it is correct?
            # TODO(gc): Is it even meaningful to padding in kv_cache
        # 2. Assemble kv_cache end

        # 3. Invoke underlying models
        bigdl_input_ids = torch.tensor(bigdl_input_ids, device=self.device)
        if is_prefill_stage:
            self.debug_print("###### In prefill stage")
            kwargs = {
                "input_ids": bigdl_input_ids,
                "attention_mask": None
                if len(bigdl_attention_mask) == 0
                else torch.tensor(bigdl_attention_mask, device=self.device),
                "past_key_values": None,
                "use_cache": True,
            }
        else:
            self.debug_print("###### In decoding stage")
            kwargs = {
                "input_ids": bigdl_input_ids,
                "past_key_values": bigdl_kv_cache_list,
                "use_cache": True,
            }
        outputs = self.model.forward(**kwargs)
        # 3. Invoke underlying models end

        # 4. Update kv_cache
        last_kv_cache = outputs.past_key_values
        # As described in documentations: here is the format of the past_key_values
        # (tuple(tuple(torch.FloatTensor))
        # The length is config.n_layers.  The embedded tensor shape:
        # ((batch_size, num_heads, sequence_length, embed_size_per_head)x2)
        # We need to maintain the KV_CACHE for each of the sequence

        # Our maintained KV_CACHE format:
        # KV_CACHE first dimension num_layers
        # KV_CACHE second dimension 2, one for key, one for values
        # KV_CACHE third layer, a dict seq_id -> torch.Tensor
        for layer in range(num_layers):
            for kv in range(2):
                batch_dim = 0
                for seq_id in processed_seq_ids:
                    # self.debug_print("past_key_values's shape: ", last_kv_cache[layer][kv].shape)
                    kv_cache[layer][kv][seq_id] = last_kv_cache[layer][kv][batch_dim]
                    batch_dim += 1
        # 4. Update kv_cache ends

        # 5. applying sampler
        # Find the last token for each batch
        # Size (B, seq_len, embedding_len)
        logits = outputs.logits[:, -1, :]

        # TODO(gc): fix sampler, remove timestamp
        bigdl_output = self.sampler(logits, input_metadata, time.perf_counter())
        # 5. applying sampler ends

        # 6. return result to invoker
        return bigdl_output
