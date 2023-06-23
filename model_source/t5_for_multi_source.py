import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5Config, T5PreTrainedModel, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5DenseActDense, T5DenseGatedActDense, T5LayerFF, T5Attention, T5LayerSelfAttention, T5LayerCrossAttention, T5Stack
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions, ModelOutput
import copy
from typing import Optional, Tuple, Union, Any, Dict
from dataclasses import dataclass, fields
from collections import OrderedDict, UserDict
import warnings
import gc
import inspect
import json
import os


class T5BlockDecoder(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = True
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,

        encoder_hidden_states_1=None,
        encoder_attention_mask_1=None,
        encoder_hidden_states_2=None,
        encoder_attention_mask_2=None,

        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states_1 is None else 6

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'4 (past / key) for cross attention x 2. ' if expected_num_past_key_values == 6 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value_1 = past_key_value[2:4]
            cross_attn_past_key_value_2 = past_key_value[4:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value_1, cross_attn_past_key_value_2 = None, None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)


        # 1st cross attention
        do_cross_attention = self.is_decoder and encoder_hidden_states_1 is not None and encoder_hidden_states_2 is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            #-----------------1st cross attention------------------------------
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states_1,
                attention_mask=encoder_attention_mask_1,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value_1,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            #-----------------2st cross attention------------------------------

            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[2](
                hidden_states,
                key_value_states=encoder_hidden_states_2,
                attention_mask=encoder_attention_mask_2,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value_2,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
    
    
    
    
class T5StackDecoder(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5BlockDecoder(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states_1=None,
        encoder_attention_mask_1=None,
        encoder_hidden_states_2=None,
        encoder_attention_mask_2=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        # if self.model_parallel:
        #     torch.cuda.set_device(self.first_device)
        #     self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask_1 is None and encoder_hidden_states_1 is not None and encoder_attention_mask_2 is None and encoder_hidden_states_2 is not None:
            encoder_seq_length_1 = encoder_hidden_states_1.shape[1]
            encoder_attention_mask_1 = torch.ones(batch_size, encoder_seq_length_1, device=inputs_embeds.device, dtype=torch.long)

            encoder_seq_length_2 = encoder_hidden_states_2.shape[1]
            encoder_attention_mask_2 = torch.ones(batch_size, encoder_seq_length_2, device=inputs_embeds.device, dtype=torch.long)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states_1 is not None and encoder_hidden_states_2 is not None:
            encoder_batch_size_1, encoder_sequence_length_1, _ = encoder_hidden_states_1.size()
            encoder_hidden_shape_1 = (encoder_batch_size_1, encoder_sequence_length_1)
            if encoder_attention_mask_1 is None:
                encoder_attention_mask_1 = torch.ones(encoder_hidden_shape_1, device=inputs_embeds.device)
            encoder_extended_attention_mask_1 = self.invert_attention_mask(encoder_attention_mask_1)

            encoder_batch_size_2, encoder_sequence_length_2, _ = encoder_hidden_states_2.size()
            encoder_hidden_shape_2 = (encoder_batch_size_2, encoder_sequence_length_2)
            if encoder_attention_mask_2 is None:
                encoder_attention_mask_2 = torch.ones(encoder_hidden_shape_2, device=inputs_embeds.device)
            encoder_extended_attention_mask_2 = self.invert_attention_mask(encoder_attention_mask_2)
        else:
            encoder_extended_attention_mask_1 = None
            encoder_extended_attention_mask_2 = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            # if self.model_parallel:
            #     torch.cuda.set_device(hidden_states.device)
            #     # Ensure that attention_mask is always on the same device as hidden_states
            #     if attention_mask is not None:
            #         attention_mask = attention_mask.to(hidden_states.device)
            #     if position_bias is not None:
            #         position_bias = position_bias.to(hidden_states.device)
            #     if encoder_hidden_states is not None:
            #         encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            #     if encoder_extended_attention_mask is not None:
            #         encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            #     if encoder_decoder_position_bias is not None:
            #         encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            #     if layer_head_mask is not None:
            #         layer_head_mask = layer_head_mask.to(hidden_states.device)
            #     if cross_attn_layer_head_mask is not None:
            #         cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,

                    encoder_hidden_states_1,
                    encoder_extended_attention_mask_1,

                    encoder_hidden_states_2,
                    encoder_extended_attention_mask_2,

                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,

                    encoder_hidden_states_1 = encoder_hidden_states_1,
                    encoder_attention_mask_1 = encoder_extended_attention_mask_1,

                    encoder_hidden_states_2 = encoder_hidden_states_2,
                    encoder_attention_mask_2 = encoder_extended_attention_mask_2,

                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states_1 is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            # if self.model_parallel:
            #     for k, v in self.device_map.items():
            #         if i == v[-1] and "cuda:" + str(k) != self.last_device:
            #             hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )



# @add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5ForMultiSourceConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder_1 = T5Stack(encoder_config, self.shared)
        self.encoder_2 = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5StackDecoder(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None



    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder_1.set_input_embeddings(new_embeddings)
        self.encoder_2.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    # def get_encoder(self):
        # return self.encoder_1, self.encoder_2
        # return self.encoder_1
    
    def get_encoder_output(self, encoder_kwargs):
        input_ids = encoder_kwargs['input_ids']
        attention_mask = encoder_kwargs['attention_mask']

        input_ids_1 =  torch.stack((list(map(lambda x:x[:input_ids.shape[1]//2] , input_ids))), dim=0)
        input_ids_2 =  torch.stack((list(map(lambda x:x[input_ids.shape[1]//2:] , input_ids))), dim=0)
        attention_mask_1 =  torch.stack((list(map(lambda x:x[:attention_mask.shape[1]//2] , attention_mask))), dim=0)
        attention_mask_2 =  torch.stack((list(map(lambda x:x[attention_mask.shape[1]//2:] , attention_mask))), dim=0)

        encoder_kwargs_1 = copy.deepcopy(encoder_kwargs)
        encoder_kwargs_2 = copy.deepcopy(encoder_kwargs)

        encoder_kwargs_1['input_ids'] = input_ids_1
        encoder_kwargs_1['attention_mask'] = attention_mask_1

        encoder_kwargs_2['input_ids'] = input_ids_2
        encoder_kwargs_2['attention_mask'] = attention_mask_2

        encoder_outputs_1 = self.encoder_1(**encoder_kwargs_1)
        encoder_outputs_2 = self.encoder_2(**encoder_kwargs_2)


        if(encoder_outputs_1.past_key_values or encoder_outputs_1.attentions or encoder_outputs_1.cross_attentions or 
           encoder_outputs_2.past_key_values or encoder_outputs_2.attentions or encoder_outputs_2.cross_attentions):
           raise ValueError("past_key_values=None, attentions=None, cross_attentions=None are defined")


        encoder_outputs = copy.deepcopy(encoder_outputs_2)
        encoder_outputs['last_hidden_state'] = torch.cat((encoder_outputs_1.last_hidden_state, encoder_outputs_2.last_hidden_state), dim=1)
        if(encoder_outputs_1.hidden_states):
            encoder_outputs['hidden_states'] = tuple(map(lambda x:torch.cat((encoder_outputs_1.hidden_states[x], encoder_outputs_2.hidden_states[x]), dim=1), 
                                               list(range(len(encoder_outputs_1.hidden_states)))))

        return encoder_outputs

    # def get_decoder(self):
    #     return self.decoder

    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        # input_ids_1: Optional[torch.LongTensor] = None,
        # attention_mask_1: Optional[torch.FloatTensor] = None,
        # input_ids_2: Optional[torch.LongTensor] = None,
        # attention_mask_2: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,

        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,

        # encoder_outputs_1: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        # encoder_outputs_2: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""


        input_ids_1 = None
        input_ids_2 = None
        if(input_ids is not None):
            input_ids_1 =  torch.stack((list(map(lambda x:x[:input_ids.shape[1]//2] , input_ids))), dim=0)
            input_ids_2 =  torch.stack((list(map(lambda x:x[input_ids.shape[1]//2:] , input_ids))), dim=0)

        attention_mask_1 = None
        attention_mask_2 = None
        if(attention_mask is not None):
            attention_mask_1 =  torch.stack((list(map(lambda x:x[:attention_mask.shape[1]//2] , attention_mask))), dim=0)
            attention_mask_2 =  torch.stack((list(map(lambda x:x[attention_mask.shape[1]//2:] , attention_mask))), dim=0)

        encoder_outputs_1 = None
        encoder_outputs_2 = None
        if(encoder_outputs is not None):
            encoder_outputs_1 = copy.deepcopy(encoder_outputs)
            encoder_outputs_2 = copy.deepcopy(encoder_outputs)
            encoder_outputs_1['last_hidden_state'] = torch.stack((list(map(lambda x:x[:encoder_outputs[0].shape[1]//2] , encoder_outputs[0]))), dim=0)
            encoder_outputs_2['last_hidden_state'] = torch.stack((list(map(lambda x:x[encoder_outputs[0].shape[1]//2:] , encoder_outputs[0]))), dim=0)


        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs_1 is None and encoder_outputs_2 is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs_1 = self.encoder_1(
                input_ids=input_ids_1,
                attention_mask=attention_mask_1,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            encoder_outputs_2 = self.encoder_2(
                input_ids=input_ids_2,
                attention_mask=attention_mask_2,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if(encoder_outputs_1.past_key_values or encoder_outputs_1.attentions or encoder_outputs_1.cross_attentions or 
               encoder_outputs_2.past_key_values or encoder_outputs_2.attentions or encoder_outputs_2.cross_attentions):
               raise ValueError("past_key_values=None, attentions=None, cross_attentions=None are defined")

            encoder_outputs_hidden_states = None
            if(encoder_outputs_1.hidden_states or encoder_outputs_1.hidden_states):
                encoder_outputs_hidden_states = tuple(map(lambda x:torch.cat((encoder_outputs_1.hidden_states[x], encoder_outputs_2.hidden_states[x]), dim=1), 
                                                             list(range(len(encoder_outputs_1.hidden_states)))))
                

            encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=torch.cat((encoder_outputs_1.last_hidden_state, encoder_outputs_1.last_hidden_state), dim=1),
                past_key_values=None, 
                hidden_states=encoder_outputs_hidden_states, 
                attentions=None, 
                cross_attentions=None)

        elif return_dict and not isinstance(encoder_outputs_1, BaseModelOutput) and not isinstance(encoder_outputs_2, BaseModelOutput):
            encoder_outputs_1 = BaseModelOutput(
                last_hidden_state=encoder_outputs_1[0],
                hidden_states=encoder_outputs_1[1] if len(encoder_outputs_1) > 1 else None,
                attentions=encoder_outputs_1[2] if len(encoder_outputs_1) > 2 else None,
            )
            encoder_outputs_2 = BaseModelOutput(
                last_hidden_state=encoder_outputs_2[0],
                hidden_states=encoder_outputs_2[1] if len(encoder_outputs_2) > 1 else None,
                attentions=encoder_outputs_2[2] if len(encoder_outputs_2) > 2 else None,
            )
            # print('encoder_outputs_1',encoder_outputs_1)
        hidden_states_1 = encoder_outputs_1[0]
        hidden_states_2 = encoder_outputs_2[0]
        
        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)
        #     hidden_states_1 = hidden_states_1.to(self.decoder.first_device)
        #     hidden_states_2 = hidden_states_2.to(self.decoder.first_device)
        #     if decoder_input_ids is not None:
        #         decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        #     if attention_mask_1 is not None:
        #         attention_mask_1 = attention_mask_1.to(self.decoder.first_device)
        #     if attention_mask_2 is not None:
        #         attention_mask_2 = attention_mask_2.to(self.decoder.first_device)
        #     if decoder_attention_mask is not None:
        #         decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)


        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states_1=hidden_states_1,
            encoder_attention_mask_1=attention_mask_1,
            encoder_hidden_states_2=hidden_states_2,
            encoder_attention_mask_2=attention_mask_2,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.encoder.first_device)
        #     self.lm_head = self.lm_head.to(self.encoder.first_device)
        #     sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output



        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,

            encoder_last_hidden_state = encoder_outputs.last_hidden_state,
            encoder_hidden_states = encoder_outputs.hidden_states,
            encoder_attentions = encoder_outputs.attentions
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. get encoder
        # encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        # encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_signature = {'return_dict', 'output_attentions', 'use_cache', 'cross_attn_head_mask', 'encoder_hidden_states', 'encoder_attention_mask', 'input_ids', 'past_key_values', 'inputs_embeds', 'attention_mask', 'output_hidden_states', 'head_mask'}
        # print('encoder_signature',encoder_signature)

        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        # print('encoder_kwargs', encoder_kwargs)
        # print(encoder_kwargs['input_ids'].shape)
        # print(encoder_kwargs['attention_mask'].shape)

        # model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        # print(' model_kwargs["encoder_outputs"]',  model_kwargs["encoder_outputs"])

        # encoder_output = self.get_encoder_output(encoder_kwargs)
        # model_kwargs["encoder_outputs_1"] = encoder_output['encoder_outputs_1']
        # model_kwargs["encoder_outputs_2"] = encoder_output['encoder_outputs_2']

        model_kwargs["encoder_outputs"]: ModelOutput = self.get_encoder_output(encoder_kwargs)

        return model_kwargs
    



class CustomDatasetForMultiSource(torch.utils.data.Dataset):
    def __init__(self, tokenizer, text_data_1, text_data_2, labels):
        self.text_data_1 = text_data_1
        self.text_data_2 = text_data_2
        self.labels = labels
        self.source_len = 512
        self.summ_len = 100
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_data_1)

    def __getitem__(self, idx):
        text_1 = self.text_data_1[idx]
        text_2 = self.text_data_2[idx]
        label = self.labels[idx]

        # Tokenize text inputs
        text_input_1 = self.tokenizer.batch_encode_plus([text_1], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        text_input_2 = self.tokenizer.batch_encode_plus([text_2], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([label], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        input_ids_1 = text_input_1['input_ids'].squeeze()
        attention_mask_1 = text_input_1['attention_mask'].squeeze()
        input_ids_2 = text_input_2['input_ids'].squeeze()
        attention_mask_2 = text_input_2['attention_mask'].squeeze()

        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        input_ids = torch.cat((input_ids_1, input_ids_2), dim=1)
        attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=1)

        # return {
        #     'input_ids_1': input_ids_1.to(dtype=torch.long), 
        #     'attention_mask_1': attention_mask_1.to(dtype=torch.long), 
        #     'input_ids_2': input_ids_2.to(dtype=torch.long), 
        #     'attention_mask_2': attention_mask_2.to(dtype=torch.long), 
        #     'target_ids': target_ids.to(dtype=torch.long),
        #     'target_ids_y': target_ids.to(dtype=torch.long)
        # }

        return {
            'input_ids': input_ids.to(dtype=torch.long), 
            'attention_mask': attention_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }