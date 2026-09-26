import json


class JaumeConfig:
    model_type = "jaume"

    def __init__(
        self,
        vocab_size: int = 151646,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        max_seq_len: int = 8192,
        expert_num: int = 4,
        topk: int = 2,
        mlp_bias: bool = False,
        use_moe: bool = True,
        rope_theta: float = 1000000.0,
        embedding_dim: int = 512,
        matryoshka_dims: tuple = (128, 256, 384, 512),
        pooling: str = "attention",
        pad_token_id: int = 151643,
        load_balance_weight: float = 0.01,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_seq_len = max_seq_len
        self.expert_num = expert_num
        self.topk = topk
        self.mlp_bias = mlp_bias
        self.use_moe = use_moe
        self.rope_theta = rope_theta
        self.embedding_dim = embedding_dim
        self.matryoshka_dims = tuple(matryoshka_dims)
        self.pooling = pooling
        self.pad_token_id = pad_token_id
        self.load_balance_weight = load_balance_weight

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        d = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        d["matryoshka_dims"] = list(self.matryoshka_dims)
        d["model_type"] = self.model_type
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d.pop("model_type", None)
        d["matryoshka_dims"] = tuple(d.get("matryoshka_dims", (128, 256, 384, 512)))
        return cls(**d)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
