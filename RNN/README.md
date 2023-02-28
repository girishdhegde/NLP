# RNN
* neural network for processing **sequential data**, such as time-series data or natural language text. 
* have kind of **loops** within their architecture ->  **remember information across time-steps**. 
* maintain and update **hidden states** -> kind of **memory**.
* **vanishing gradients** -> short term memory.
* pseudo code
    ```python
    RNNCell
        # Initializations
        i2h = LinearLayer(input_size, hidden_size)
        h2h = LinearLayer(hidden_size, hidden_size)
        h2o = LinearLayer(hidden_size, hidden_size)

        # Forward pass
        hidden = i2h(input) + h2h(hidden)
        output = h2o(h)
        hidden, ouput = nonlinearity(hidden), nonlinearity(output)
    ```


# Character Level RNN
* **each character** in the input sequence is represented as a **one-hot** vector
* which is then fed into the network **one at a time**
* The network processes each character in turn, using its internal state to **maintain a memory** of the **previous characters** in the sequence
* Output at each time step is a probability distribution over the possible characters that could come **next in the sequence**

# Codes Implemented
* CharRNN
* Character level sequence tokenization
* char2char and chars2class custom pytorch dataloader
* Variable sequence length collate function
* BiGram visualization
* Character level sampling from trained model

# CharRNN Results
* Names Sampled from model trained on dataset of names
  ```python
  kailanie, keilee, marlanie, meik, ariahna, kaileigh, analie, mariana, aleah, million, kairon, mariamandri, malayaan, kiman, marielle, kaylin, arlis, kayline, anan, annalyn, alaina, kelanie, annika, analia, kennet, alice, alexa, alissa, arlen, kailon, andreya, kairo, kaisen, alissandra, michel, marlen, kairon, maryshia, alicia, michel, marylynn, analea, alexia, annabella, ariely, alexxa, arias, marlana, alanie, kingson, arianna, mikeem, alania, karlie, karin, kayla, kennadee, maken, kara, analeigh, analeigh, kayleen, alice, aryah, aliana, marlayah, alaysha, mila, maryanne, karina, karam, makeyla, aleaha, melia, arisha, alekai, alianna, aryaan, kelani, alisha, analisa, analia, kaileena, marlin, arlee, makariah, malayah, mikiyah, analeigha, malikai, kennede, alisandro, kaylin, andrey, aryna, andersen, kennesse, marielle, alaniel, analea
  ```
* BiGram Visualization
<p align="center">
  <img src="./assets/bigram.png" alt="bigram">
  <br>
  <em></em>
</p>

# References
* https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
* http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
* https://pytorch.org/docs/stable/generated/torch.nn.RNN.html