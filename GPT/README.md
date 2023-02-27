# GPT - Generative Pre-Trainig Transformer

* **decoder only causal** transformer
* **sub-word** level **byte-pair-encoding** tokenizer.
* **pre-trained** on a large corpus of text data using a **self-supervised causal language modelling** i.e. next token prediction
* can be **fine-tuned** on a specific **downstream** task by adding  **task-specific output layer** or **task-specific tokens** and **training** the model on a **labeled** dataset.

* **coding-task** finetuning
    ```python
    Input:
        <|question_token|> question <|code_token|> code <|end|>
    Target:
        ignore_id................................. code <|end|>
    ```

# Codes Implemented
* Causal Decoder only Transformer - GPT
* Optimizer with decay and no decay params groups.
* BPE subword level tokenizer
* Gradient Accumulation training
* Self supervised pre-training
* coding task finetuning


# References
* https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
* https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
* https://github.com/karpathy/minGPT
* https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0