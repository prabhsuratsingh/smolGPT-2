from collections import deque
from functools import lru_cache
import json
from typing import Counter



class BPE:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.bpe_ranks = {}
    
    def train(
            self,
            text,
            vocab_size,
            allowed_special={"<|endoftext|>"}
    ):
        processed = []

        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed.append("Ġ")
            if char != " ":
                processed.append(char)
        processed = "".join(processed)

        unq_chars = [chr(i) for i in  range(256)]
        unq_chars.extend(
            char for char in sorted(set(processed))
            if char not in unq_chars
        )
        if "Ġ" not in unq_chars:
            unq_chars.append("Ġ")

        self.vocab = {i : char for i, char in enumerate(unq_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id
        
        token_ids = [self.inverse_vocab[char] for char in processed]

        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            
            token_ids = self.replace_pairs(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id
        
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def encode(self, text, allowed_special=None):
        import re

        token_ids = []

        if allowed_special is not None and len(allowed_special) > 0:
            special_pattern = (
                "(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")"
            )

            last_idx = 0

            for match in re.finditer(special_pattern, text):
                prefix = text[last_idx:match.start()]
                token_ids.extend(self.encode(prefix, allowed_special=None))

                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocab")
                last_idx = match.end()
            
            text = text[last_idx:]

            disallowed = [
                tok for tok in self.inverse_vocab
                if tok.startswith("<|") and tok.endswith("|>") and tok in text and tok not in allowed_special
            ]

            if disallowed:
                raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
        
        tokens = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")
            words = line.split()
            for j, word in enumerate(words):
                if j == 0 and i > 0:
                    tokens.append("Ġ" + word)
                elif j == 0:
                    tokens.append(word)
                else:
                    tokens.append("Ġ" + word)
        
        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))
        
        return token_ids
    
    def tokenize_with_bpe(self, token):
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.bpe_merges:
                    merged_token_id = self.bpe_merges[pair]
                    new_tokens.append(merged_token_id)
                    i += 2  
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens
        return token_ids

    def decode(self, token_ids):
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " " 
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string
    
    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id



    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None
        
        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid Mode!!")
    
    @staticmethod
    def replace_pairs(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            curr = dq.popleft()
            if dq and (curr, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(curr)
        
        return replaced


if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPE()
    tokenizer.train(
        text,
        vocab_size=1000,
        allowed_special={"<|endoftext|>"}
    )

    # print("VOCAB Size : ")
    # print(len(tokenizer.vocab))
    # print("BPE Merges Size : ")
    # print(len(tokenizer.bpe_merges))

    # input_text = "Jack embraced beauty through art and life."
    # token_ids = tokenizer.encode(input_text)
    # print(token_ids)

    # print(tokenizer.decode(
    #     tokenizer.encode("This is some text.")
    # ))

    tokenizer.save_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")