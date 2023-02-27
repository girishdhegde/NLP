# LSTM - Long Short Term Memory
* RNN that handles **long-term** dependencies. 
* use **gates** to control the flow of information through the network.
* **selectively remember or forget** information as needed.
* maintain and update extra **cell state** in controlled manner through gates.
* disadvatages: **slower** to compute, **diffuclt** to **parallelize**, limited context, **limited interpretability** vs tranformers.
* pseudo code
    ```python
    LSTMCell
        i = sigmoid(wii.x + bii + whi.h + bhi) - input gate.
        f = sigmoid(wif.x + bif + whf.h + bhf) - forget gate.
        o = sigmoid(wio.x + bio + who.h + bho) - output gate.
        g = tanh(wig.x + big + whg.h + bhg) - candidiates.
        c = f*c + i*g - updated cell state.
        h = o*tanh(c) - updated hidden/outout state.
    ```

# Codes Implemented
* Word Level LSTM
* Word level sequence tokenization
* Text Corpus next word prediction pytorch dataloader
* BiLSTM - LSTM1(sequence) + LSTM2(reversed(sequence))
* Embedding visualization
* Neuron Firing Visualizatoin

# References
* https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf
* https://colah.github.io/posts/2015-08-Understanding-LSTMs/
* https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html