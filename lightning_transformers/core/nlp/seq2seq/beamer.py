import math
import random
from dataclasses import dataclass
from logging import Logger
from typing import TYPE_CHECKING, Optional, Tuple, Type

import torch
from hydra.utils import get_class
from torch import nn
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqModelOutput
from transformers.models.blenderbot.configuration_blenderbot import BlenderbotConfig
from transformers.models.blenderbot.modeling_blenderbot import (
    BlenderbotAttention,
    BlenderbotDecoder,
    BlenderbotDecoderLayer,
    BlenderbotEncoder,
    BlenderbotEncoderLayer,
    BlenderbotForConditionalGeneration,
    BlenderbotLearnedPositionalEmbedding,
    BlenderbotModel,
    _expand_mask,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from lightning_transformers.core.config import LitTaskConfig, OptimizerConfig, SchedulerConfig
from lightning_transformers.core.instantiator import Instantiator
from lightning_transformers.core.nlp.config import HFBackboneConfig
from lightning_transformers.core.nlp.model import HFTransformer
from lightning_transformers.core.nlp.seq2seq.config import HFSeq2SeqConfig
from lightning_transformers.core.nlp.seq2seq.model import Seq2SeqTransformer

if TYPE_CHECKING:
    from transformers import AutoModel


class Seq2SeqBeamer(Seq2SeqTransformer):
    def __init__(
        self,
        downstream_model_type: str,
        backbone: HFBackboneConfig,
        optimizer: OptimizerConfig = OptimizerConfig(),
        scheduler: SchedulerConfig = SchedulerConfig(),
        instantiator: Optional[Instantiator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        pipeline_kwargs: Optional[dict] = None,
        cfg: Optional[LitTaskConfig] = None,
        # seq2seq_cfg: HFSeq2SeqConfig = HFSeq2SeqConfig(),
        **model_data_kwargs,
    ) -> None:
        self.save_hyperparameters()
        # self.cfg = seq2seq_cfg
        model_cls: Type["AutoModel"] = get_class(downstream_model_type)
        config = AutoConfig.from_pretrained(backbone.pretrained_model_name_or_path, **model_data_kwargs)
        model = BeamerForConditionalGeneration(config, cfg)
        blenderbot = model_cls.from_pretrained(backbone.pretrained_model_name_or_path, **model_data_kwargs)
        res = model.load_state_dict(blenderbot.state_dict(), strict=False)
        print(f'The following parameters are missing from the state_dict: {res.missing_keys}')
        super(HFTransformer, self).__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator, cfg=cfg)
        self._tokenizer = tokenizer  # necessary for hf_pipeline
        self._hf_pipeline = None
        self._hf_pipeline_kwargs = pipeline_kwargs or {}


class BeamerForConditionalGeneration(BlenderbotForConditionalGeneration):
    def __init__(self, config: BlenderbotConfig, rnn_config):
        super(BlenderbotForConditionalGeneration, self).__init__(config)
        self.model = BeamerModel(config, rnn_config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()


class BeamerModel(BlenderbotModel):
    def __init__(self, config: BlenderbotConfig, rnn_config):
        super(BlenderbotModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        if rnn_config.rnn_share:
            shared_rnn = BeamerRNN(config, batch_first=True, rnn_class=rnn_config.rnn_class)
        else:
            shared_rnn = None
        self.encoder = BeamerEncoder(config, rnn_config, self.shared, shared_rnn)
        self.decoder = BeamerDecoder(config, rnn_config, self.shared, shared_rnn)

        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
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
            encoder_rnn_states=encoder_outputs.rnn_states,
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


class BeamerEncoder(BlenderbotEncoder):
    def __init__(self, config: BlenderbotConfig, rnn_config, embed_tokens: Optional[nn.Embedding] = None, shared_rnn=None):
        super(BlenderbotEncoder, self).__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        if shared_rnn is None:
            shared_rnn = BeamerRNN( # the RNN model that shared among all layers
                config,
                batch_first=True,
                rnn_class=rnn_config.rnn_class,
            )
        self.layers = nn.ModuleList([BeamerEncoderLayer(config, shared_rnn) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        rnn_states = None
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        rnn_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        rnn_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                rnn_states = layer_outputs[-1]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BeamerBaseOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states,
            attentions=all_attentions, rnn_states=rnn_states,
        )


class BeamerDecoder(BlenderbotDecoder):
    def __init__(self, config: BlenderbotConfig, rnn_config, embed_tokens: Optional[nn.Embedding] = None, shared_rnn=None):
        super(BlenderbotDecoder, self).__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        if shared_rnn is None:
            shared_rnn = BeamerRNN( # the RNN model that shared among all layers
                config,
                batch_first=True,
                rnn_class=rnn_config.rnn_class,
            )
        self.layers = nn.ModuleList([BeamerDecoderLayer(config, shared_rnn) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.alignment_attn = AlignmentAttention(config.d_model, config.encoder_ffn_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_rnn_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # project encoder rnn_states to decoder size
        if encoder_rnn_states is not None:
            rnn_states = self.alignment_attn(inputs_embeds, encoder_rnn_states, encoder_attention_mask.squeeze(1))
        else:
            rnn_states = None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    Logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    rnn_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    rnn_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            rnn_states = layer_outputs[-1]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)

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


class BeamerEncoderLayer(BlenderbotEncoderLayer):
    def __init__(self, config: BlenderbotConfig, shared_rnn):
        super(BlenderbotEncoderLayer, self).__init__()
        self.embed_dim = config.d_model
        self.self_attn = BlenderbotAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.rnn = shared_rnn

    def forward(
        self,
        hidden_states: torch.Tensor,
        rnn_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, rnn_states = self.rnn(hidden_states, rnn_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        
        outputs += (rnn_states,) # don't disturb other fields' reserved positions

        return outputs


class BeamerDecoderLayer(BlenderbotDecoderLayer):
    def __init__(self, config: BlenderbotConfig, shared_rnn):
        super(BlenderbotDecoderLayer, self).__init__()
        self.embed_dim = config.d_model

        self.self_attn = BlenderbotAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BlenderbotAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.rnn = shared_rnn

    def forward(
        self,
        hidden_states: torch.Tensor,
        rnn_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

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

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, rnn_states = self.rnn(hidden_states, rnn_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)
        
        outputs += (rnn_states,)

        return outputs


class BeamerRNN(nn.Module):
    "A wrapper class for rnn. Conducts reshape before and after."

    RNN_OPTS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}

    def __init__(self, args, batch_first=True, rnn_class='rnn'):
        super(BeamerRNN, self).__init__()
        # if is_decoder:
        #     self.emb_size = args.d_model
        #     self.hid_size = args.encoder_ffn_dim
        # else:
        assert args.encoder_ffn_dim == args.decoder_ffn_dim
        self.emb_size = args.d_model
        self.hid_size = args.encoder_ffn_dim
        self.num_layers = 1
        self.batch_first = batch_first

        rnn_class = BeamerRNN.RNN_OPTS[rnn_class]

        self.rnn = rnn_class(
            self.emb_size,
            self.hid_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=args.dropout if self.num_layers > 1 else 0.0,
        )
        # TODO: self.linear = nn.Linear(hiddensize, embedding_size)
        # nn.init.xavier_uniform_(self.rnn.weight)

    def forward(self, x, hid=None):
        # flatten input and hid
        if self.batch_first:
            input = x.view(-1, 1, self.emb_size)
        else:
            input = x.view(1, -1, self.emb_size)
        if hid is not None:
            if type(self.rnn) is nn.LSTM:
                hid = (
                    hid[0].view(self.num_layers, -1, self.hid_size),
                    hid[1].view(self.num_layers, -1, self.hid_size)
                )
            else:
                hid = hid.view(self.num_layers, -1, self.hid_size)

        # feed into rnn
        out, hid = self.rnn(input, hid.contiguous() if hid is not None else hid)

        # reshape back
        if type(self.rnn) is nn.LSTM:
            hid = (
                hid[0].view(self.num_layers, x.size(0), x.size(1), self.hid_size),
                hid[1].view(self.num_layers, x.size(0), x.size(1), self.hid_size),
            )
        else:
            out = out.view(x.size(0), x.size(1), self.hid_size)
            hid = hid.view(self.num_layers, x.size(0), x.size(1), self.hid_size)
        return out, hid


@dataclass
class BeamerBaseOutput(BaseModelOutput):
    rnn_states: Optional[Tuple[torch.FloatTensor]] = None


class AlignmentAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        rnn_state_dim: int,
        # dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        # self.dropout = dropout

        # only project key vectors from rnn_state_dim to embed_dim
        self.k_proj = nn.Linear(rnn_state_dim, embed_dim, bias=bias)
    
    def forward(
        self,
        query,
        key_value_states,
        # past_key_states: Optional[torch.Tensor] = None, # used during inference to avoid redundant computing
        attention_mask: Optional[torch.Tensor] = None,
    ):
        key = self.k_proj(key_value_states)
        
        tgt_seq_len, emb_sz = query.size()[-2:]
        src_seq_len = key.size(2)
        rnn_state_size = key_value_states.size(-1)
        query = query.repeat(key.size(0), 1, 1, 1)
        attn_weights = torch.bmm(query.view(-1, tgt_seq_len, emb_sz), key.view(-1, src_seq_len, emb_sz).transpose(1, 2))
        attn_weights = attn_weights.reshape(key.size(0), key.size(1), tgt_seq_len, src_seq_len)
        if attention_mask is not None:
            attn_weights += attention_mask.repeat(key.size(0), 1, 1, 1)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_weights.view(-1, tgt_seq_len, src_seq_len), key_value_states.view(-1, src_seq_len, rnn_state_size))
        
        return attn_output.view(key.size(0), key.size(1), tgt_seq_len, rnn_state_size)
