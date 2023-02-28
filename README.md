# NLP
This repository contains implementation of Deep learning based **Language Models** from scratch. Including:
* [RNN](./RNN)
* [LSTM](./LSTM)
* [Transformer](./Transformer)
* [GPT](./GPT)


## Neural language modeling
Neural language modeling involves using neural networks to model the probability distribution of the words in a sequence. 
* **CLM** - Causal Language Modelling: Unidirectional next token prediction, Used to generate fluent texts.
* **MLM** - Masked Language Modelling: Masked tokens prediction, Used to learn good representations.

# Getting Started

```shell
git clone https://github.com/girishdhegde/NLP.git
```

## Requirements
* python >= 3.9.13
* pytorch >= 1.13.1

## Installation
```
    pip install -r requirements.txt
```

# Usage
## Project Structure
```bash
NLP
├── Algorithm[RNN/LSTM/Transformer/GPT]
│    ├── data.py - tokenizer, dataset, dataloader, collate_fn
│    ├── utils.py - save/load ckpt, log prediction, sample from model
│    ├── algorithm.py - model
│    ├── train.py - training loop
│    ├── demo.py - inference
│    └── {something}_viz.py - visualization

```
## Model Import
```python
# from ALGORITHM.algorithm import Model
from GPT.gpt import GPT

net = GPT(
    emb_dim, heads, num_layers,
    vocab_size, context_size,
    emb_dropout=0.1, attn_dropout=0.1, res_dropout=0.1,
)
```

## Training and Inference
* train.py in each algorith subdirectory has training code.
* edit the **ALL_CAPITAL** parameters section at the starting of train.py as required. 
* demo.py in each algorithm subdirectory has inference code.

## Note
Refer respective README.md in subdirectories for more details.

## License - MIT
# References
* https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
* http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
* https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
* https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf
* https://colah.github.io/posts/2015-08-Understanding-LSTMs/
* https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
* https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
* https://github.com/lucidrains/x-transformers
* https://nlp.seas.harvard.edu/2018/04/03/attention.html
* http://peterbloem.nl/blog/transformers
* https://jalammar.github.io/illustrated-transformer/
* https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
* https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
* https://github.com/karpathy/minGPT
* https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0