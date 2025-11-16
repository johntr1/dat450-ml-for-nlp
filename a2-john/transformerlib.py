
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
import torch.nn.functional as F


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # TODO: initalize components here
        self.V = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.W = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.W2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=False)
        self.Swish_1 = nn.SiLU()

    def forward(self, hidden_states):
        xW = self.W(hidden_states)
        xV = self.V(hidden_states)

        x = self.Swish_1(xW) * xV
        x = self.W2(x) 
        return x


# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        # TODO: initalize weights here
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
    
    def forward(self, hidden_states):
        ...


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.hidden_size % self.num_heads == 0
        
        self.head_dim = self.hidden_size // self.num_heads

        # TODO: set up W_q, W_k, W_v, W_o here
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # TODO: set up normalizers here
        self.q_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def _reshape_to_heads(self, x):
        #transform to (batch_size, num_heads, seq_length, head_dim)
        batch_size, seq_length, dim = x.shape
        heads = self.num_heads
        head_dim = self.head_dim
        return x.view(batch_size, seq_length, heads, head_dim).transpose(1, 2)
    
    def _merge_heads(self, x):
        #transform back to (batch_size, seq_length, hidden_size)
        batch_size, heads, seq_length, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch_size, seq_length, heads * head_dim)
        
    def forward(self, hidden_states, rope_rotations):
        
        q = self.W_q(hidden_states)
        k = self.W_k(hidden_states)
        v = self.W_v(hidden_states)

        #rmsnorm
        q = self.q_norm(q)
        k = self.k_norm(k)  
        
        # split into heads
        q = self._reshape_to_heads(q)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)
        
        # applying rope to q, k
        q, k = apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        
        attn_output = self._merge_heads(attn_output) # (batch_size, seq_length, hidden_size)

        out = self.W_o(attn_output)
        return out

class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # TODO: set up attention, MLP, and normalizers here.
        self.hidden_size = config.hidden_size

        self.attn = A2Attention(config)

        self.mlp = A2MLP(config)
        
        self.attn_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, rope_rotations):
        attn_out = self.attn(hidden_states, rope_rotations)
        attn_out = self.attn_norm(attn_out)

        hidden_states = hidden_states + attn_out

        mlp_out = self.mlp(hidden_states)
        mlp_out = self.mlp_norm(mlp_out)

        hidden_states = hidden_states + mlp_out
        return hidden_states

        


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        
        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # TODO: put all transformer decoder layers in a ModuleList.
        self.layers = nn.ModuleList(
            [A2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers
        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.

        hidden_states = self.embedding(input_ids)  # (batch_size, seq_length, hidden_size) for debug
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_rotations)
        
        hidden_states = self.final_norm(hidden_states)

        logits = self.unembedding(hidden_states)  # (batch_size, seq_length, vocab_size)
        return logits
        

#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin

if __name__ == "__main__":
    class DummyConfig:
        vocab_size = 1000
        hidden_size = 64
        intermediate_size = 256
        num_attention_heads = 8
        num_hidden_layers = 2
        rms_norm_eps = 1e-6

    cfg = A2ModelConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=8,
        num_hidden_layers=2,
        rope_theta=100000.0,
        rms_norm_eps=1e-6
    )
    model = A2Transformer(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 10))   # (B, T)
    logits = model(x)
    print("Logits shape:", logits.shape)  # (2, 10, 1000)