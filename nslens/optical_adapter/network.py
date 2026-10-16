"""Optical embeddings -> frozen transformer -> continuous, bounded action.

No vocabulary logits, text decoding, or detached hidden-state cache. Gradients
must flow THROUGH the frozen backbone to train its optical input adapters.
"""
import json
import hashlib
from pathlib import Path
import torch
from torch import nn
from .state import CELL_DIM, GLOBAL_DIM, PROMPT, SCHEMA
from nslens.spectrometer import model


class OpticalEncoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(CELL_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.global_state = nn.Sequential(nn.Linear(GLOBAL_DIM, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, cells, state):
        return self.cell(cells), self.global_state(state)


class NumericPolicy(nn.Module):
    """Small numerical-only control, with the same observations and action bounds."""
    def __init__(self, width=64):
        super().__init__()
        self.encoder = OpticalEncoder(width)
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 36))
        self.metadata = dict(kind='numeric', width=width, pretrained=False)

    def forward(self, cells, state):
        c, g = self.encoder(cells, state)
        return self.head(torch.cat([c.mean(dim=1), g], dim=-1)).tanh()


class QwenOpticalPolicy(nn.Module):
    def __init__(self, backbone, prefix_ids, width=64, metadata=None):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        hidden = backbone.config.hidden_size
        self.encoder = OpticalEncoder(width)
        self.cell_projection = nn.Linear(width, hidden)
        self.global_projection = nn.Linear(width, hidden)
        self.action_token = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.normal_(self.action_token, std=.02)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.SiLU(), nn.Linear(width, 36))
        # Fixed prefix, no padding; every batch member shares this task in v1.
        self.register_buffer('prefix_ids', torch.as_tensor(prefix_ids, dtype=torch.long).reshape(1, -1))
        self.metadata = dict(metadata or {}, width=width, hidden_size=hidden)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen dropout behavior must remain deterministic
        return self

    def forward(self, cells, state):
        batch = cells.shape[0]
        c, g = self.encoder(cells, state)
        prefix = self.backbone.get_input_embeddings()(self.prefix_ids.expand(batch, -1))
        tokens = torch.cat([prefix, self.global_projection(g)[:, None].to(prefix.dtype),
                            self.cell_projection(c).to(prefix.dtype),
                            self.action_token.expand(batch, -1, -1).to(prefix.dtype)], dim=1)
        # Deliberately no no_grad(): optical projections need gradients through Qwen.
        result = self.backbone(inputs_embeds=tokens, use_cache=False,
                               attention_mask=torch.ones(tokens.shape[:2], dtype=torch.long, device=tokens.device))
        return self.head(result.last_hidden_state[:, -1].float()).tanh()


def tiny_qwen(seed=17, width=64):
    """Actual Qwen3.5 hybrid layers at reduced size, RANDOM weights; never pretrained."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, layer_types=['linear_attention', 'full_attention'],
        max_position_embeddings=256,
        rope_parameters=dict(rope_type='default', rope_theta=10000., partial_rotary_factor=1.,
                             mrope_section=[2, 3, 3], mrope_interleaved=True))
    config._attn_implementation = 'eager'
    return QwenOpticalPolicy(Qwen3_5TextModel(config), [1, 2, 3], width,
        dict(kind='tiny_qwen', pretrained=False, seed=seed, config=config.to_dict(),
             prefix='three synthetic task tokens; not a language-understanding experiment'))


def pretrained_qwen(weights, device='cuda:0', width=64):
    """Load the locally cached Qwen3.5-family checkpoint, then bypass vision/LM head.

    v1 uses a single GPU. Refuse an underprovisioned device before loading; never
    evict services, silently offload, or substitute a different pretrained model.
    """
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    weights = Path(weights).resolve()
    config = json.loads((weights / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_5':
        raise ValueError('This loader supports the inspected qwen3_5 checkpoint architecture only')
    if not device.startswith('cuda:') or not torch.cuda.is_available():
        raise ValueError('The full checkpoint path requires an explicitly selected CUDA device')
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    required = sum(p.stat().st_size for p in weights.glob('*.safetensors')) + 8 * 1024**3
    if free < required:
        raise RuntimeError(f'Insufficient free GPU memory: {free/1024**3:.1f} GiB; '
                           f'conservative loading/training floor {required/1024**3:.1f} GiB. '
                           'Leave existing workloads intact and retry on a free GPU.')
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    ids = tokenizer.encode(PROMPT, add_special_tokens=True)
    full = AutoModelForImageTextToText.from_pretrained(weights, local_files_only=True,
        device_map={'': device}, dtype=torch.bfloat16, attn_implementation='sdpa')
    backbone = full.model.language_model
    backbone.config.use_cache = False
    policy = QwenOpticalPolicy(backbone, ids, width, dict(kind='pretrained_qwen', pretrained=True,
        weights=str(weights), model_type=config['model_type'], prompt=PROMPT,
        quantization=config.get('quantization_config'), device=device))
    # Move only new float32 modules; never cast/move the quantized backbone.
    for module in (policy.encoder, policy.cell_projection, policy.global_projection, policy.head):
        module.to(device)
    policy.action_token.data = policy.action_token.data.to(device)
    policy.prefix_ids = policy.prefix_ids.to(device)
    del full
    return policy


def save_adapter(policy, path):
    """Never duplicate the large frozen backbone in adapter checkpoints."""
    weights = {k: v.detach().cpu() for k, v in policy.state_dict().items() if not k.startswith('backbone.')}
    torch.save(dict(metadata=policy.metadata, weights=weights, schema=SCHEMA,
        simulator_sha256=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()), path)


def load_adapter(policy, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema') != SCHEMA or saved.get('simulator_sha256') != hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest():
        raise ValueError('Feature schema or optical simulator changed')
    expected_metadata = dict(policy.metadata); stored_metadata = dict(saved['metadata'])
    expected_metadata.pop('device', None); stored_metadata.pop('device', None)
    if stored_metadata != expected_metadata:
        raise ValueError('Adapter/backbone provenance mismatch')
    expected = {k for k in policy.state_dict() if not k.startswith('backbone.')}
    if set(saved['weights']) != expected:
        raise ValueError('Incomplete or unexpected adapter tensors')
    policy.load_state_dict(saved['weights'], strict=False)
    return policy
