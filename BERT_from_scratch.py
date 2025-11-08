# Mini BERT-style Encoder (MLM + NSP) trained on WikiText-2

# This script implements an encoder-only Transformer from scratch in PyTorch
# with BERT-style embeddings, self-attention encoder stack, an MLM head, and an NSP head.
#
# Key features:
# - Token + Position + Segment (token_type) embeddings
# - Multi-head self-attention encoder layers with Feed-Forward networks
# - [CLS] and [SEP] special tokens usage like BERT
# - MLM head with weight tying to token embeddings
# - NSP head operating on pooled [CLS]
# - BERT-style masking strategy: 80% [MASK], 10% random token, 10% unchanged
# - Simple sentence-pair dataset builder for NSP
#
# Usage (example):
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#   pip install datasets transformers
#   python mini_bert_wikitext2.py --device cpu --epochs 2 --batch_size 16
#
# Notes:
# - You can switch to GPU by setting --device cuda (if available).

import math
import os
import random
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import BertTokenizerFast



# Utility: GELU (BERT)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)



# Config

@dataclass
class MiniBertConfig:
    vocab_size: int
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2  # sentence A/B
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0



# Embeddings

class BertEmbeddings(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



# Multi-Head Self-Attention

class BertSelfAttention(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (B, H, T, D)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (B, T, C)
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Attention scores: (B, H, T, T)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask expected shape (B, 1, 1, T) with 0 for keep, -inf for mask
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # (B, H, T, D)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)  # (B, T, C)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states: (B, T, C); take [CLS] at position 0
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



# BERT-style Encoder Model

class MiniBertForPreTraining(nn.Module):
    def __init__(self, config: MiniBertConfig):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        # MLM head
        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_act = GELU()
        self.mlm_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))

        # NSP head
        self.nsp_classifier = nn.Linear(config.hidden_size, 2)

        # Tie weights: decoder shares with embeddings
        self.mlm_decoder.weight = self.embeddings.word_embeddings.weight

    def forward(self, input_ids, token_type_ids, attention_mask):
        # Build extended attention mask: (B, 1, 1, T) with 0 for keep, -inf for mask
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e4

        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(sequence_output)

        # MLM head
        prediction_scores = self.mlm_dense(sequence_output)
        prediction_scores = self.mlm_act(prediction_scores)
        prediction_scores = self.mlm_layer_norm(prediction_scores)
        prediction_scores = self.mlm_decoder(prediction_scores) + self.mlm_bias

        # NSP head
        seq_relationship_score = self.nsp_classifier(pooled_output)

        return prediction_scores, seq_relationship_score



# Data: Sentence splitting and example building

SENT_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def naive_sent_tokenize(text: str) -> List[str]:
    # Simple, fast sentence splitter as a fallback (you can swap for nltk.sent_tokenize if desired)
    # Splits on periods, exclamation marks, question marks followed by whitespace.
    # Removes very short fragments.
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 1]
    return sents


class WikiText2NSPMLMDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: BertTokenizerFast,
        max_length: int = 128,
        mask_prob: float = 0.15,
        rng_seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.rng = random.Random(rng_seed)

        # Load and preprocess sentences
        ds = load_dataset("wikitext", "wikitext-2-v1", split=split)
        raw_texts = ds["text"]

        # Filter out headings and empty lines, then split into sentences
        docs: List[List[str]] = []
        current: List[str] = []
        for line in raw_texts:
            line = line.strip()
            if not line:
                # new doc boundary
                if current:
                    docs.append(current)
                    current = []
                continue
            if line.startswith("=") and line.endswith("="):
                # skip titles
                continue
            sents = naive_sent_tokenize(line)
            current.extend(sents)
        if current:
            docs.append(current)

        # Build pairs: consecutive sentence pairs as positives; negatives = random sentence from random doc
        self.pairs: List[Tuple[str, str, int]] = []  # (A, B, is_next)
        all_sents = [s for doc in docs for s in doc]
        for doc in docs:
            for i in range(len(doc) - 1):
                a, b = doc[i], doc[i + 1]
                # Positive example
                self.pairs.append((a, b, 1))
                # Negative example: pair A with random B
                neg_b = a
                while True:
                    neg_b = self.rng.choice(all_sents)
                    if neg_b != b:
                        break
                self.pairs.append((a, neg_b, 0))

        self.vocab_size = self.tokenizer.vocab_size
        self.special_ids = set([
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.mask_token_id,
        ])

    def __len__(self):
        return len(self.pairs)

    def _apply_mlm(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Return (masked_input_ids, mlm_labels) with BERT 80/10/10 masking.
        Non-masked positions in labels are -100 so CrossEntropy ignores them.
        """
        labels = [-100] * len(input_ids)

        # candidate positions exclude specials
        candidate_pos = [i for i, tid in enumerate(input_ids) if tid not in self.special_ids]
        self.rng.shuffle(candidate_pos)
        num_to_mask = max(1, int(round(len(candidate_pos) * self.mask_prob)))
        masked_pos = set(candidate_pos[:num_to_mask])

        masked_input_ids = list(input_ids)
        for pos in masked_pos:
            original_id = input_ids[pos]
            labels[pos] = original_id
            p = self.rng.random()
            if p < 0.8:
                masked_input_ids[pos] = self.tokenizer.mask_token_id
            elif p < 0.9:
                # replace with random token (avoid special ids)
                while True:
                    rnd = self.rng.randrange(self.vocab_size)
                    if rnd not in self.special_ids:
                        break
                masked_input_ids[pos] = rnd
            else:
                # leave unchanged
                pass
        return masked_input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        a, b, is_next = self.pairs[idx]
        enc = self.tokenizer(
            a,
            b,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"]
        token_type_ids = enc["token_type_ids"]
        attention_mask = enc["attention_mask"]

        masked_input_ids, labels = self._apply_mlm(input_ids)

        return {
            "input_ids": torch.tensor(masked_input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "mlm_labels": torch.tensor(labels, dtype=torch.long),
            "nsp_label": torch.tensor(is_next, dtype=torch.long),
        }


def collate_batch(features: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    # Dynamic padding to batch max length
    max_len = max(len(f["input_ids"]) for f in features)

    def pad(seq, pad_id):
        return seq + [pad_id] * (max_len - len(seq))

    input_ids = torch.tensor([pad(f["input_ids"].tolist(), pad_token_id) for f in features], dtype=torch.long)
    token_type_ids = torch.tensor([pad(f["token_type_ids"].tolist(), 0) for f in features], dtype=torch.long)
    attention_mask = torch.tensor([pad(f["attention_mask"].tolist(), 0) for f in features], dtype=torch.long)
    mlm_labels = torch.tensor([pad(f["mlm_labels"].tolist(), -100) for f in features], dtype=torch.long)
    nsp_label = torch.tensor([f["nsp_label"].item() for f in features], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "mlm_labels": mlm_labels,
        "nsp_label": nsp_label,
    }



# Training / Evaluation


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, device, vocab_size):
    model.train()
    total_mlm, total_nsp, total = 0.0, 0.0, 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        prediction_scores, seq_relationship = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])

        mlm_loss = F.cross_entropy(
            prediction_scores.view(-1, vocab_size),
            batch["mlm_labels"].view(-1),
            ignore_index=-100,
        )
        nsp_loss = F.cross_entropy(seq_relationship, batch["nsp_label"])
        loss = mlm_loss + nsp_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_mlm += mlm_loss.item()
        total_nsp += nsp_loss.item()
        total += loss.item()

    n = len(dataloader)
    return {
        "mlm": total_mlm / n,
        "nsp": total_nsp / n,
        "total": total / n,
    }


def evaluate(model, dataloader, device, vocab_size):
    model.eval()
    total_mlm, total_nsp, total = 0.0, 0.0, 0.0
    correct_nsp, count_nsp = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            prediction_scores, seq_relationship = model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])

            mlm_loss = F.cross_entropy(
                prediction_scores.view(-1, vocab_size),
                batch["mlm_labels"].view(-1),
                ignore_index=-100,
            )
            nsp_loss = F.cross_entropy(seq_relationship, batch["nsp_label"])
            loss = mlm_loss + nsp_loss

            total_mlm += mlm_loss.item()
            total_nsp += nsp_loss.item()
            total += loss.item()

            preds = seq_relationship.argmax(dim=-1)
            correct_nsp += (preds == batch["nsp_label"]).sum().item()
            count_nsp += preds.size(0)

    n = len(dataloader)
    return {
        "mlm": total_mlm / n,
        "nsp": total_nsp / n,
        "total": total / n,
        "nsp_acc": correct_nsp / max(1, count_nsp),
    }



# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_out", type=str, default="mini_bert.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    # Tokenizer (WordPiece)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Datasets
    train_ds = WikiText2NSPMLMDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_prob=args.mask_prob,
        rng_seed=args.seed,
    )
    val_ds = WikiText2NSPMLMDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_prob=args.mask_prob,
        rng_seed=args.seed + 1,
    )

    collate = lambda batch: collate_batch(batch, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Model
    config = MiniBertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        type_vocab_size=2,
    )

    model = MiniBertForPreTraining(config)
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting training on {args.device}...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config.vocab_size)
        val_metrics = evaluate(model, val_loader, device, config.vocab_size)
        print(
            f"Epoch {epoch}: train_total={train_metrics['total']:.4f} (mlm={train_metrics['mlm']:.4f}, nsp={train_metrics['nsp']:.4f}) | "
            f"val_total={val_metrics['total']:.4f} (mlm={val_metrics['mlm']:.4f}, nsp={val_metrics['nsp']:.4f}, nsp_acc={val_metrics['nsp_acc']:.3f})"
        )

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "tokenizer": tokenizer.name_or_path,
    }, args.model_out)
    print(f"Saved to {args.model_out}")


if __name__ == "__main__":
    main()
