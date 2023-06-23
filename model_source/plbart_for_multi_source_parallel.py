import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from transformers.models.plbart.modeling_plbart import PLBartEncoder, PLBartLearnedPositionalEmbedding, PLBartAttention, _make_causal_mask, _expand_mask
from transformers import PLBartForConditionalGeneration, PLBartTokenizer, PLBartPreTrainedModel, PLBartConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqModelOutput, ModelOutput
import copy
from typing import Optional, Tuple, Union, Any
from dataclasses import dataclass, fields
from collections import OrderedDict, UserDict
import warnings
import math
import random
import numpy as np
from transformers.modeling_utils import PreTrainedModel, logging
from transformers.activations import ACT2FN
import warnings
import gc
import inspect
import json
import os

class PLBartDecoderLayer(nn.Module):
    def __init__(self, config: PLBartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = PLBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn_1 = PLBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm_1 = nn.LayerNorm(self.embed_dim)

        self.encoder_attn_2 = PLBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm_2 = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,

        encoder_hidden_states_1: Optional[torch.Tensor] = None,
        encoder_attention_mask_1: Optional[torch.Tensor] = None,

        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,

        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)


        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states_1 is not None:
            residual = hidden_states

            # --------------------------1st---------------------------------------
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None


            hidden_states_1, cross_attn_weights_1, cross_attn_present_key_value_1 = self.encoder_attn_1(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states_1,
                attention_mask=encoder_attention_mask_1,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states_1 = nn.functional.dropout(hidden_states_1, p=self.dropout, training=self.training)
            hidden_states_1 = residual + hidden_states_1
            hidden_states_1 = self.encoder_attn_layer_norm_1(hidden_states_1)

            # --------------------------1st---------------------------------------
            hidden_states_2, cross_attn_weights_2, cross_attn_present_key_value_2 = self.encoder_attn_2(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states_2,
                attention_mask=encoder_attention_mask_2,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states_2 = nn.functional.dropout(hidden_states_2, p=self.dropout, training=self.training)
            hidden_states_2 = residual + hidden_states_2
            hidden_states_2 = self.encoder_attn_layer_norm_2(hidden_states_2)

            hidden_states = torch.add(hidden_states_1, hidden_states_2)/2
            cross_attn_present_key_value = (torch.add(cross_attn_present_key_value_1[0], cross_attn_present_key_value_2[0])/2, 
                                            torch.add(cross_attn_present_key_value_1[1], cross_attn_present_key_value_2[1])/2)
            
            # hidden_states = torch.div(torch.add(hidden_states_1, hidden_states_2), 2)
            # cross_attn_present_key_value = (torch.div(torch.add(cross_attn_present_key_value_1[0], cross_attn_present_key_value_2[0]), 2), 
            #                                 torch.div(torch.add(cross_attn_present_key_value_1[1], cross_attn_present_key_value_2[1]), 2))
            # hidden_states = hidden_states_2
            # cross_attn_present_key_value = cross_attn_present_key_value_2

            
            if(cross_attn_weights_1 is not None or cross_attn_weights_2 is not None):
                raise ValueError("cross_attn_weights_1 or cross_attn_weights_2 are not None")
            else:
                cross_attn_weights = None

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    
class PLBartDecoder(PLBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`PLBartDecoderLayer`]

    Args:
        config: PLBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PLBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = PLBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([PLBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        encoder_hidden_states_1 = None
        encoder_hidden_states_2 = None
        if(encoder_hidden_states is not None):
            encoder_hidden_states_1 =  torch.stack((list(map(lambda x:x[:encoder_hidden_states.shape[1]//2] , encoder_hidden_states))), dim=0)
            encoder_hidden_states_2 =  torch.stack((list(map(lambda x:x[encoder_hidden_states.shape[1]//2:] , encoder_hidden_states))), dim=0)

        encoder_attention_mask_1 = None
        encoder_attention_mask_2 = None
        if(encoder_attention_mask is not None):
            encoder_attention_mask_1 =  torch.stack((list(map(lambda x:x[:encoder_attention_mask.shape[1]//2] , encoder_attention_mask))), dim=0)
            encoder_attention_mask_2 =  torch.stack((list(map(lambda x:x[encoder_attention_mask.shape[1]//2:] , encoder_attention_mask))), dim=0)


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states_1 is not None and encoder_attention_mask_1 is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask_1 = _expand_mask(encoder_attention_mask_1, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if encoder_hidden_states_2 is not None and encoder_attention_mask_2 is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask_2 = _expand_mask(encoder_attention_mask_2, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None


            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,

                    encoder_hidden_states_1,
                    encoder_attention_mask_1,

                    encoder_hidden_states_2,
                    encoder_attention_mask_2,

                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
                print('layer_outputs',layer_outputs)
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,

                    encoder_hidden_states_1 = encoder_hidden_states_1,
                    encoder_attention_mask_1 = encoder_attention_mask_1,

                    encoder_hidden_states_2 = encoder_hidden_states_2,
                    encoder_attention_mask_2 = encoder_attention_mask_2,

                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
        
# @add_start_docstrings("The bare PLBART Model outputting raw hidden-states without any specific head on top.",PLBART_START_DOCSTRING)
class PLBartModel(PLBartPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: PLBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder_1 = PLBartEncoder(config, self.shared)
        self.encoder_2 = PLBartEncoder(config, self.shared)

        self.decoder = PLBartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder_1.embed_tokens = self.shared
        self.encoder_2.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder_1(self):
        return self.encoder_1
    def get_encoder_2(self):
        return self.encoder_2

    # def get_encoder(self):
    #     return self.encoder_1

    # def get_decoder(self):
    #     return self.decoder

    # @add_start_docstrings_to_model_forward(PLBART_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:


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



        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, PLBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)


        # print('encoder_outputs',encoder_outputs)
        if encoder_outputs_1 is None and encoder_outputs_2 is None:
            encoder_outputs_1 = self.encoder_1(
                input_ids=input_ids_1,
                attention_mask=attention_mask_1,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            encoder_outputs_2 = self.encoder_2(
                input_ids=input_ids_2,
                attention_mask=attention_mask_2,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # print('encoder_outputs_1', encoder_outputs_1)

            encoder_outputs = BaseModelOutput(
                last_hidden_state = torch.cat((encoder_outputs_1['last_hidden_state'], encoder_outputs_2['last_hidden_state']), dim=1),
                hidden_states = None,
                attentions =  None,
            )


        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
# @add_start_docstrings("The PLBART Model with a language modeling head. Can be used for code-to-text, text-to-code and code-to-code.",PLBART_START_DOCSTRING)
class PLBartForMultiSourceParallelConditionalGeneration(PLBartPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder.version",
        r"decoder.version",
        r"lm_head.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_tokens.weight",
    ]


    def __init__(self, config: PLBartConfig):
        super().__init__(config)
        self.model = PLBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    # def get_encoder(self):
    #     return self.model.get_encoder()

    # def get_decoder(self):
    #     return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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

        encoder_outputs_1 = self.model.get_encoder_1()(**encoder_kwargs_1)
        encoder_outputs_2 = self.model.get_encoder_2()(**encoder_kwargs_2)

        encoder_outputs_last_hidden_state = torch.cat((encoder_outputs_1[0], encoder_outputs_2[0]), dim=1)
        encoder_outputs = copy.deepcopy(encoder_outputs_2)
        encoder_outputs['last_hidden_state'] = encoder_outputs_last_hidden_state

        return encoder_outputs


    # @add_start_docstrings_to_model_forward(PLBART_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # @add_end_docstrings(PLBART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        **kwargs,  # TODO: Check if this is needed. It is unused?
    ) -> Dict[str, Any]:
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


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
        # print('encoder_signature',encoder_signature)
        encoder_signature = {'return_dict', 'head_mask', 'attention_mask', 'output_attentions', 'output_hidden_states', 'inputs_embeds', 'input_ids'}

        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        # model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        # print(' model_kwargs["encoder_outputs"]',  model_kwargs["encoder_outputs"])

        # encoder_output = self.get_encoder_output(encoder_kwargs)
        # model_kwargs["encoder_outputs_1"] = encoder_output['encoder_outputs_1']
        # model_kwargs["encoder_outputs_2"] = encoder_output['encoder_outputs_2']

        model_kwargs["encoder_outputs"]: ModelOutput = self.get_encoder_output(encoder_kwargs)

        return model_kwargs
    
