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

# Results
* **22M** parameters gpt-2 like model
* **Pre-Training**
    * **40M** tokens dataset
    * **100K** iterations
* Code **Fine-Tuning**
    * 3 Epochs
    * **4000** samples
* Code Generation(Next Token Prediction)
```python
<|INPUT|>
Sinchan and his friends loves to eat. They have a 2D rectangular cake which they want to share. Sinchan is very kind and offers his friends to eat the cake first, all friends goes one by one to eat the cake. Each friend will cut cake into two parts. First part of the cake will be largest possible square from the cake, that the friend will eat, and will leave the second part for others, it continues untill cake is over or every friend gets to eat the cake.
Now Sinchan wonder how much cake would he be able to get.

-----Input-----
- First line of input contain T, denoting number of test case.
- First line of each test case contains N, denoting number of friends.
- Second line of test case contain L, B (L x B) denoting dimension of cake.

-----Output-----
- For each test case, if Sinchan gets to eat the cake print (without quotes) "Yes #" where # is area of cake that Sinchan gets. Otherwise print  (without quotes) "No"

-----Constraints-----
- 1 ≤ T, N, L, B ≤ 1000

-----Example-----
Input:
3
2
5 3
4
4 8
1
1 2

Output:
Yes 2
No
Yes 1

-----Explanation-----
    Example case 1. First friend divides 5x3 cake in 3x3 and 2x3 since 3x3 is largest possible square, second Friend divides 2x3 cake in 2x2 and 1x2 and eat 2x2 piece. Finaly Sinchan gets to eat and area of cake is 1*2 hence Output is (without quotes) "Yes 2"

<|PREDICTION|>
for _ in range(eval(input())):
    n = eval(input())
    [l,b] = list(map(int,input().split()))
    for i in range(n):
        if l > b :
            l -= b
        elif b > l:
            b -= l 
        else:
            l=b=0
            break
        if b== 0 or l==0:
            break
    if b == 0 or l == 0:
        print('No')
    else:
        print('Yes',l*b)<|10|> 
```
# References
* https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
* https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
* https://github.com/karpathy/minGPT
* https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0