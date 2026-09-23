"""
tokenalloc/encoder.py
─────────────────────
Transmitter steps 1-2 (CLAUDE.md §4): task tokens + masked BERT encoder.

Input sequence:   [CLS] S1..S4 F1..F4 E1..E4 <words> [SEP]

Attention rules (query -> key):
  - word tokens ([CLS], words, [SEP]) attend only to word tokens
      => word processing is exactly as in pretraining, and CLS carries no task info
  - task tokens attend to word tokens and to their OWN task's tokens
  - task tokens of different tasks never attend to each other

Position ids: task tokens get position 0 and words keep their original positions
1..n, so the word stream is bit-for-bit the stream BERT saw in pretraining.

Outputs kept: CLS [B, H] and task tokens [B, T, J, H]. Word outputs are discarded.
"""

import torch
import torch.nn as nn


class TaskTokenEncoder(nn.Module):
    def __init__(self, bert: nn.Module, n_tasks: int = 3, tokens_per_task: int = 4):
        super().__init__()
        self.bert = bert
        self.n_tasks = n_tasks
        self.tokens_per_task = tokens_per_task
        self.hidden = bert.config.hidden_size
        n = n_tasks * tokens_per_task
        self.task_tokens = nn.Parameter(torch.randn(n, self.hidden) * 0.02)
        # task id of every task-token slot: [0,0,0,0,1,1,1,1,2,2,2,2]
        self.register_buffer("slot_task", torch.arange(n_tasks).repeat_interleave(tokens_per_task),
                             persistent=False)

    @property
    def n_slots(self):
        return self.n_tasks * self.tokens_per_task

    def build_attention(self, word_mask: torch.Tensor) -> torch.Tensor:
        """
        word_mask [B, L] (1 = real word token incl. CLS/SEP). Returns a boolean
        [B, L+S, L+S] matrix, True where query i may attend key j. Order of the
        sequence: CLS, S task tokens, then word positions 1..L-1.
        """
        B, L = word_mask.shape
        S = self.n_slots
        dev = word_mask.device
        is_word = torch.cat([torch.ones(1, dtype=torch.bool, device=dev),
                             torch.zeros(S, dtype=torch.bool, device=dev),
                             torch.ones(L - 1, dtype=torch.bool, device=dev)])
        valid = torch.cat([word_mask[:, :1].bool(), torch.ones(B, S, dtype=torch.bool, device=dev),
                           word_mask[:, 1:].bool()], dim=1)                        # [B, L+S]
        task_of = torch.full((L + S,), -1, device=dev)
        task_of[1:1 + S] = self.slot_task
        same_task = (task_of[:, None] == task_of[None, :]) & (task_of[:, None] >= 0)  # [L+S, L+S]
        key_is_valid_word = (is_word[None, :] & valid)                              # [B, L+S]
        allow = key_is_valid_word[:, None, :].expand(B, L + S, L + S).clone()
        allow = allow | same_task[None]
        # padding queries: let them see CLS only (their outputs are never used)
        pad_q = ~valid
        allow[pad_q] = False
        allow[:, :, 0] |= pad_q
        return allow

    def _run_bert(self, embeds, position_ids, allow):
        kwargs = dict(inputs_embeds=embeds, position_ids=position_ids,
                      token_type_ids=torch.zeros_like(position_ids))
        try:                                     # transformers >= 5: boolean 4D mask
            return self.bert(attention_mask=allow[:, None], **kwargs).last_hidden_state
        except (ValueError, RuntimeError, NotImplementedError):
            # transformers 4.x: 3D {0,1} mask, broadcast over heads internally
            return self.bert(attention_mask=allow.long(), **kwargs).last_hidden_state

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_words=False):
        B, L = input_ids.shape
        S = self.n_slots
        words = self.bert.embeddings.word_embeddings(input_ids)                    # [B, L, H]
        tasks = self.task_tokens.unsqueeze(0).expand(B, S, self.hidden)
        embeds = torch.cat([words[:, :1], tasks, words[:, 1:]], dim=1)            # [B, L+S, H]
        pos = torch.cat([torch.zeros(S + 1, dtype=torch.long, device=input_ids.device),
                         torch.arange(1, L, device=input_ids.device)])
        position_ids = pos.unsqueeze(0).expand(B, L + S)
        out = self._run_bert(embeds, position_ids, self.build_attention(attention_mask))
        cls = out[:, 0]
        task_out = out[:, 1:1 + S].reshape(B, self.n_tasks, self.tokens_per_task, self.hidden)
        if return_words:
            return cls, task_out, torch.cat([out[:, :1], out[:, 1 + S:]], dim=1)
        return cls, task_out


@torch.no_grad()
def check_attention_rules(enc: TaskTokenEncoder, input_ids, attention_mask, atol=1e-4):
    """
    Verifies the masks really act inside BERT (transformers versions differ in how
    they read custom masks). Perturbs the task-token embeddings of one task and
    checks that (a) word and CLS outputs do not move and (b) other tasks' tokens
    do not move, while (c) that task's own tokens do.
    """
    was_training = enc.training
    enc.eval()
    cls0, t0, w0 = enc(input_ids, attention_mask, return_words=True)
    saved = enc.task_tokens.data.clone()
    J = enc.tokens_per_task
    g = torch.Generator().manual_seed(0)                              # perturb task 0 only
    enc.task_tokens.data[:J] += torch.randn(J, enc.hidden, generator=g).to(saved.device)
    cls1, t1, w1 = enc(input_ids, attention_mask, return_words=True)
    enc.task_tokens.data.copy_(saved)
    enc.train(was_training)
    valid = attention_mask.bool()
    word_diff = (w0 - w1).abs()[valid].max().item()
    cls_diff = (cls0 - cls1).abs().max().item()
    other_diff = (t0[:, 1:] - t1[:, 1:]).abs().max().item()
    own_diff = (t0[:, 0] - t1[:, 0]).abs().max().item()
    ok = word_diff < atol and cls_diff < atol and other_diff < atol and own_diff > atol
    return ok, {"word_diff": word_diff, "cls_diff": cls_diff,
                "other_task_diff": other_diff, "own_task_diff": own_diff}
