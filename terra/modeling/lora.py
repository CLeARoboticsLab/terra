from terra.modeling import modules
from flax import nnx
from typing import Sequence


def add_lora_to_transformer(
    transformer: modules.TransformerEncoder | modules.TransformerDecoder,
    rngs: nnx.Rngs,
    add_to: Sequence[str] = ("mlp",),
    lora_rank: int = 8,
):
    for layer in transformer.layers:
        add_lora_to_transformer_layer(layer, rngs, add_to=add_to, lora_rank=lora_rank)


def add_lora_to_transformer_layer(
    transformer_layer: (
        modules.TransformerEncoderLayer | modules.TransformerDecoderLayer
    ),
    rngs: nnx.Rngs,
    add_to: Sequence[str] = ("mlp",),
    lora_rank: int = 8,
):
    for t in add_to:
        if t == "mlp":
            add_lora_to_mlp(transformer_layer.dense_proj, lora_rank, rngs)
        else:
            raise ValueError(f"Unknown module type to add LoRA to: {t}")


def add_lora_to_mlp(mlp: modules.MLP, rngs: nnx.Rngs, lora_rank: int = 8):
    layers = []
    for layer in mlp.layers:
        layers.append(_lora_wrap_linear(layer, lora_rank, rngs))
    mlp.layers = layers
    mlp.output_layer = _lora_wrap_linear(mlp.output_layer, lora_rank, rngs)


def _lora_wrap_linear(linear: nnx.Linear, rngs: nnx.Rngs, lora_rank: int = 8):
    return nnx.LoRA(
        in_features=linear.in_features,
        lora_rank=lora_rank,
        out_features=linear.out_features,
        base_module=linear,
        rngs=rngs,
    )
