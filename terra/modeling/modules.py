import flax.nnx as nnx
import jax.numpy as jnp
import jax
from typing import List


class TransformerEncoderLayer(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        mask: str | None = None,
        **kwargs,
    ):
        self.mask = mask

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            dropout_rate=attention_dropout_rate,
            decode=False,
            deterministic=deterministic,
            rngs=rngs,
        )

        self.dense_proj = MLP(
            input_dim=embed_dim,
            hidden_dims=[mlp_hidden_dim],
            output_dim=embed_dim,
            rngs=rngs,
            dropout_rate=mlp_dropout_rate,
            deterministic=deterministic,
        )

        self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, inputs, mask=None, deterministic: bool | None = None):
        if mask is not None:
            padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32)
        elif self.mask == "causal" or mask == "causal":
            padding_mask = nnx.make_causal_mask(inputs[:, :, 0])
        else:
            padding_mask = None

        attention_output = self.attention(
            inputs_q=inputs,
            inputs_k=inputs,
            inputs_v=inputs,
            mask=padding_mask,
            decode=False,
            deterministic=deterministic,
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input, deterministic=deterministic)
        return self.layernorm_2(proj_input + proj_output)


class TransformerEncoder(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        mask: str | None = None,
        **kwargs,
    ):
        self.layers = [
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                num_heads=num_heads,
                attention_dropout_rate=attention_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
                deterministic=deterministic,
                mask=mask,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x, mask=None, deterministic: bool | None = None):
        for layer in self.layers:
            x = layer(x, mask=mask, deterministic=deterministic)
        return x


class TransformerDecoderLayer(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        mask: str | None = "causal",
        **kwargs,
    ):
        self.mask = mask

        self.attention_1 = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            dropout_rate=attention_dropout_rate,
            deterministic=deterministic,
            decode=False,
            rngs=rngs,
        )
        self.attention_2 = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            dropout_rate=attention_dropout_rate,
            deterministic=deterministic,
            decode=False,
            rngs=rngs,
        )

        self.dense_proj = MLP(
            input_dim=embed_dim,
            hidden_dims=[mlp_hidden_dim],
            output_dim=embed_dim,
            rngs=rngs,
            dropout_rate=mlp_dropout_rate,
            deterministic=deterministic,
        )
        self.layernorm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layernorm_3 = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(
        self,
        inputs,
        encoder_outputs,
        mask=None,
        deterministic: bool | None = None,
    ):
        if self.mask == "causal":
            causal_mask = nnx.make_causal_mask(inputs[:, :, 0])
        else:
            causal_mask = None

        if mask is not None:
            padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            inputs_q=inputs,
            inputs_v=inputs,
            inputs_k=inputs,
            mask=causal_mask,
            deterministic=deterministic,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            inputs_q=out_1,
            inputs_v=encoder_outputs,
            inputs_k=encoder_outputs,
            mask=padding_mask,
            deterministic=deterministic,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2, deterministic=deterministic)
        return self.layernorm_3(out_2 + proj_output)


class TransformerDecoder(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
        attention_dropout_rate: float = 0.0,
        mlp_dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        mask: str | None = "causal",
        **kwargs,
    ):
        self.layers = [
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                num_heads=num_heads,
                attention_dropout_rate=attention_dropout_rate,
                mlp_dropout_rate=mlp_dropout_rate,
                deterministic=deterministic,
                mask=mask,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self, x, encoder_outputs, mask=None, deterministic: bool | None = None
    ):
        for layer in self.layers:
            x = layer(x, encoder_outputs, mask=mask, deterministic=deterministic)
        return x


class Embedding(nnx.Module):
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        hidden_dims: List[int],
        embed_dim: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
    ):

        self.enc = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=embed_dim,
            rngs=rngs,
            dropout_rate=dropout_rate,
            deterministic=deterministic,
        )

        self.pos_emb = nnx.Param(
            jax.random.normal(rngs["params"](), (seq_len, embed_dim)) * 0.02
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool | None = None):
        x = self.enc(x, deterministic=deterministic)
        x += self.pos_emb
        return x


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
    ):
        self.layers: List[nnx.Linear] = []
        self.dropout_layers: List[nnx.Dropout] = []
        prev_dim = input_dim  # Track input size for each layer
        self.output_dim = output_dim

        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(
                nnx.Linear(in_features=prev_dim, out_features=hidden_dim, rngs=rngs)
            )
            prev_dim = hidden_dim
            if i < len(hidden_dims) - 1:
                self.dropout_layers.append(
                    nnx.Dropout(
                        rate=dropout_rate, deterministic=deterministic, rngs=rngs
                    )
                )

        self.output_layer = nnx.Linear(
            in_features=prev_dim, out_features=output_dim, rngs=rngs
        )

    def __call__(self, x, deterministic: bool | None = None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = nnx.gelu(x)
            if i < len(self.layers) - 1:
                x = self.dropout_layers[i](x, deterministic=deterministic)
        x = self.output_layer(x)
        return x
