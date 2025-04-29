# Numpy Transformer Model
This project implements a Transformer model from scratch using only NumPy. The implementation focuses on understanding the core architecture of Transformers by building the model components manually without relying on high-level deep learning frameworks.

## Architecture Details
The Transformer is initialized with the following hyperparameters:
* **Embedding Dimension (d_model):** 768
* **Number of Attention Heads (num_heads):** 12
* **Feedforward Dimension (d_ff):** 3072
* **Sequence Length (seq_len):** 512

These default values are based on the BERT model architecture.

## Key Components
```positional_encoding:``` Generates sine/cosine-based encodings to inject position information into the token embeddings.

```_multi_head_attention:``` Implements scaled dot-product attention with softmax normalization.

```_feed_forward:``` A 2-layer MLP with ReLU activation.

```_layer_norm:``` Applies mean-variance normalization across the feature dimension.

```forward:``` Runs the input through the full transformer block (attention + FFN)

## How to Run
In a new python file, run the following code:
```
import numpy as np
from transformer import Transformer

x = np.random.rand(2, 512, 768)  # test input (batch_size=2)
model = Transformer()
output = model.forward(x)

print(output.shape)  # Expected output: (2, 512, 768)
```
## References

[Medium](https://medium.com/@hhpatil001/transformers-from-scratch-in-simple-python-part-i-b290760c1040)  
[Pylessons](https://pylessons.com/transformers-introduction)  
[Machine Learning Mastery](https://machinelearningmastery.com/the-transformer-model/)

## Video Walkthrough
Watch a video of me explaining the code [here](https://drive.google.com/file/d/1Xg-mITGe_EB8KKG6TmjJjsDdT4i7x75l/view?usp=sharing)
