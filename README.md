# GPT-1 from scratch 

Implemented GPT-2 and Byte-Pair Encoder from scratch using PyTorch. 

---

GPT-2 Research Paper : 
[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Byte-Pair Encoding Reference :
[Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

Hyperparameters :
```python
    vocab_size = len(tokenizer.vocab)

    model = GPT2(
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        src_pad_index=tokenizer.get_special_token_id("<|endoftext|>"),
        target_pad_index=tokenizer.get_special_token_id("<|endoftext|>"),
        embed_size=256,
        num_layers=6,
        heads=8,
        max_length=128
    ).to(device)

    dataset = GPT2Dataset(
        text=text,
        tokenizer=tokenizer,
        block_size=128
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )
```

Optimizer :
```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

Loss Function :
```python
    criterion = nn.CrossEntropyLoss()
```

Parameters :
```
Total parameters: 5,028,840
Trainable parameters: 5,028,840
```

Trained on :
```
NVIDIA GeForce GTX 1650
4GB VRAM
CUDA Version: 12.5
```