
import re
import torch
from torch.utils.data import Dataset
import pandas as pd


# ---------------------------------------------------------------------------
# Shortcut token masking (CDA probe)
# Replaces country/institutional/month tokens with [MASK] before tokenisation,
# forcing the model to attend to legal content rather than geographic proxies.
# ---------------------------------------------------------------------------

_SHORTCUT_PATTERN = re.compile(
    r'\b('
    # Month names
    r'january|february|march|april|may|june|july|august|september|october|november|december'
    # Russia
    r'|russia|russian|moscow|chelyabinsk|novosibirsk|tula|saratov|perm|volgograd|kazan|ufa'
    # Turkey
    r'|turkey|turkish|ankara|istanbul|izmir|bursa|adana|diyarbakir|gaziantep'
    # UK
    r'|britain|british|england|english|wales|welsh|scotland|scottish'
    r'|london|edinburgh|cardiff|manchester|birmingham|liverpool'
    # Poland
    r'|poland|polish|warsaw|krakow|gdansk|wroclaw|poznan|lodz'
    # Romania
    r'|romania|romanian|bucharest|cluj|timisoara|iasi|constanta|craiova'
    # Germany
    r'|germany|german|berlin|hamburg|munich|cologne|frankfurt|stuttgart|dresden'
    # France
    r'|france|french|paris|marseille|lyon|toulouse|bordeaux|lille|montpellier'
    # Other ECHR states
    r'|moldova|moldovan|chisinau'
    r'|ukraine|ukrainian|kyiv|kiev|kharkiv|odessa|lviv'
    r'|austria|austrian|vienna|graz|linz|salzburg'
    r'|belgium|belgian|brussels|antwerp|ghent|liege'
    r'|netherlands|dutch|amsterdam|rotterdam|utrecht'
    r'|sweden|swedish|stockholm|gothenburg|malmo'
    r'|norway|norwegian|oslo|bergen|trondheim'
    r'|greece|greek|athens|thessaloniki'
    r'|spain|spanish|madrid|barcelona|seville'
    r'|italy|italian|rome|milan|naples|turin'
    r'|switzerland|swiss|zurich|geneva|bern|berne|basel'
    r'|hungary|hungarian|budapest'
    # Generic ECHR/geographic
    r'|strasbourg'
    r')\b',
    re.IGNORECASE,
)

def mask_shortcuts(text: str, mask_token: str = '[MASK]') -> str:
    """Replace shortcut tokens (country/place/month names) with mask_token."""
    return _SHORTCUT_PATTERN.sub(mask_token, text)


# ---------------------------------------------------------------------------
# Reasoning-span detection
# Evaluative language tokens that mark legal reasoning in FACTS sections.
# Used by head_reasoning_tail truncation to surface the most informative span.
# ---------------------------------------------------------------------------

_REASONING_TOKENS = {
    'whether', 'particular', 'circumstances', 'legislative', 'provision',
    'provisions', 'meaning', 'within', 'obligation', 'obligations',
    'guaranteed', 'fair', 'fairness', 'impartial', 'independent',
    'reasonable', 'justif', 'proportionate', 'assess', 'assessment',
    'examine', 'examination', 'consider', 'consideration',
}

def _reasoning_score(text: str) -> float:
    """Count reasoning-token hits per word (density)."""
    words = text.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if any(w.startswith(t) for t in _REASONING_TOKENS))
    return hits / len(words)


class ECHRDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, truncation='head',
                 mask_shortcuts=False, idf_dict=None, default_idf=None):
        """
        Args:
            data:            file path (str) or pandas DataFrame
            tokenizer:       HuggingFace tokenizer
            max_len:         maximum sequence length (default 512)
            truncation:      'head'               — keep first max_len tokens
                             'head_tail'          — first 128 + last (max_len-130)
                             'head_reasoning_tail'— first 128 + best reasoning
                                                    span 128 + last (max_len-258)
                             'tfidf_sentence'     — select sentences by mean IDF
                                                    score, preserving document order
            mask_shortcuts:  if True, replace country/place/month tokens with
                             [MASK] before tokenisation (CDA debiasing)
            idf_dict:        word → IDF weight mapping (required for tfidf_sentence)
            default_idf:     IDF for OOV words; defaults to max(idf_dict.values())
        """
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.reset_index(drop=True)
        self.tokenizer      = tokenizer
        self.max_len        = max_len
        self.truncation     = truncation
        self.mask_shortcuts = mask_shortcuts
        self.idf_dict       = idf_dict or {}
        self.default_idf    = default_idf if default_idf is not None else (
            max(self.idf_dict.values()) if self.idf_dict else 1.0
        )

    def __len__(self):
        return len(self.data)

    def _preprocess(self, text: str) -> str:
        if self.mask_shortcuts:
            text = mask_shortcuts(text, self.tokenizer.mask_token or '[MASK]')
        return text

    def _encode_head_tail(self, text):
        """[CLS] first_128 last_382 [SEP]"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_head = 128
        n_tail = self.max_len - n_head - 2
        if len(tokens) <= self.max_len - 2:
            combined = tokens
        else:
            combined = tokens[:n_head] + tokens[-n_tail:]
        return self._pack(combined)

    def _encode_head_reasoning_tail(self, text):
        """[CLS] first_128 | reasoning_span_128 | last_252 [SEP]
        Splits raw text into ~200-word chunks, selects the chunk with the
        highest evaluative-token density as the 'reasoning span'.
        Falls back to head_tail if text is short enough.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_head      = 128
        n_reasoning = 128
        n_tail      = self.max_len - n_head - n_reasoning - 2  # 250

        if len(tokens) <= self.max_len - 2:
            return self._pack(tokens)

        # Split raw text into ~200-word chunks and score each
        words  = text.split()
        chunk_size = 200
        chunks = [' '.join(words[i:i + chunk_size])
                  for i in range(0, len(words), chunk_size)]
        scores = [_reasoning_score(c) for c in chunks]
        best_chunk = chunks[scores.index(max(scores))]

        reasoning_tokens = self.tokenizer.encode(
            best_chunk, add_special_tokens=False
        )[:n_reasoning]

        combined = tokens[:n_head] + reasoning_tokens + tokens[-n_tail:]
        return self._pack(combined)

    def _sentence_idf_score(self, sentence: str) -> float:
        """Mean IDF of alphabetic tokens in a sentence."""
        words = re.findall(r'[a-zA-Z]+', sentence.lower())
        if not words:
            return 0.0
        return sum(self.idf_dict.get(w, self.default_idf) for w in words) / len(words)

    def _encode_tfidf_sentence(self, text):
        """Select sentences by TF-IDF importance, reconstruct in original order.

        Strategy (ECHR best-practice):
          1. Always keep the first sentence (applicant context / narrative anchor).
          2. Score remaining sentences by mean IDF of their alphabetic tokens.
          3. Greedily add highest-scoring sentences until the token budget is full.
          4. Reassemble selected sentences in original document order so BERT
             sees a coherent narrative, not a bag of snippets.
        """
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sents:
            return self._pack([])

        budget = self.max_len - 2  # reserve [CLS] and [SEP]

        # Tokenise all sentences up front to know their lengths
        tok_ids_per_sent = [
            self.tokenizer.encode(s, add_special_tokens=False) for s in sents
        ]

        # First sentence is always selected
        selected = {0}
        used = len(tok_ids_per_sent[0])

        # Score and rank the rest
        rest_scored = sorted(
            range(1, len(sents)),
            key=lambda i: -self._sentence_idf_score(sents[i])
        )

        for i in rest_scored:
            toks = tok_ids_per_sent[i]
            if used + len(toks) <= budget:
                selected.add(i)
                used += len(toks)
            if used >= budget:
                break

        # Reconstruct in original document order
        combined = []
        for i in sorted(selected):
            combined.extend(tok_ids_per_sent[i])

        return self._pack(combined[:budget])

    def _pack(self, token_ids):
        """Add special tokens, pad to max_len, return dict of tensors."""
        input_ids = ([self.tokenizer.cls_token_id]
                     + token_ids
                     + [self.tokenizer.sep_token_id])
        # Truncate if combined exceeds max_len (can happen in reasoning path)
        input_ids = input_ids[:self.max_len]
        pad_len        = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [self.tokenizer.pad_token_id] * pad_len
        return {
            'input_ids':      torch.tensor(input_ids,      dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        text  = self._preprocess(str(row['text']))
        label = int(row['label'])

        if self.truncation == 'head_tail':
            encoding = self._encode_head_tail(text)
        elif self.truncation == 'head_reasoning_tail':
            encoding = self._encode_head_reasoning_tail(text)
        elif self.truncation == 'tfidf_sentence':
            encoding = self._encode_tfidf_sentence(text)
        else:
            enc = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            encoding = {
                'input_ids':      enc['input_ids'].flatten(),
                'attention_mask': enc['attention_mask'].flatten(),
            }

        return {
            **encoding,
            'labels': torch.tensor(label, dtype=torch.long),
        }
