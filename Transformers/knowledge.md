### Transformer networks

- A Transformer block is a family of functions $\mathbf{b} : \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$
- A Transformer is a stack of Transformer blocks: $\mathbf{f} = \mathbf{b}^{(L)} \circ \dots \circ \mathbf{b}^{(1)} : \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$
- We will use a Transformer to parameterize a NADE:
  $p_t(x_t|x_1, \dots, x_{t-1}) = \text{Cat}(\text{softmax}(\mathbf{f}_t(x_1, \dots, x_{t-1})))$
- Prefix function is defined by masking: $\mathbf{f}_t(x_1, \dots, x_{t-1}) = \mathbf{f}(\mathbf{m}_t \odot \mathbf{x})$


### Transformer networks: A high-level look

- Let us look at the model as a single black box. In a machine translation application, it would take a sentence in one language, and output its translation in another
  ![figure1](./figure/figure1.png)

- More specifically, there are an encoding component, a decoding component, and connections between them
  ![figure2](./figure/figure2.png)


### Transformer networks: A high-level look

- The encoding component is a stack of encoders. The decoding component is a stack of decoders of the same number
![figure3](./figure/figure3.png)

- The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers:
  ![figure4](./figure/figure4.png)

### Transformer networks: A high-level look
- The inputs of the encoder first flow through a self-attention layer, which helps the encoder look at other words in the input sentence as it encodes a specific word
- The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position
- The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence

  ![figure5](./figure/figure5.png)


### Bringing the tensors into the picture
- Let us start to look at the various vectors/tensors and how they flow between these components to turn the input of a trained model into an output
- As is the case in NLP applications in general, we begin by turning each input word into a vector using an embedding algorithm

  ![figure6](./figure/figure6.png)
  （Figure说明）：Each word is embedded into a vector of size 512


- The embedding only happens in the bottom-most encoder. The size of the vector is hyperparameter, typically set to the length of the longest sentence in the training dataset


### Bringing the tensors into the picture
- Each of the embedded vectors flows through each of the two layers of the encoder

  ![figure7](./figure/figure7.png)

- Here we begin to see one important property of the Transformer, which is that the word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer
- The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel while flowing through the feed-forward layer


### The residuals
- In more detail, each sub-layer (self-attention, ffn) in each encoder has a residual connection around it, and is followed by a layer-normalization step

  ![figure8](./figure/figure8.png)

- This goes for the sub-layers of the decoder as well. If we are to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:

  ![figure9](./figure/figure9.png)


### Transformer blocks
- For weights matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times k}$, and input matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$:
  1. Calculate
     $$\mathbf{Q} = \mathbf{X} \times \mathbf{W}_Q \in \mathbb{R}^{n \times k} \ (\text{queries})$$
     $$\mathbf{K} = \mathbf{X} \times \mathbf{W}_K \in \mathbb{R}^{n \times k} \ (\text{keys})$$
     $$\mathbf{V} = \mathbf{X} \times \mathbf{W}_V \in \mathbb{R}^{n \times k} \ (\text{values})$$
  2. Calculate
     $$\mathbf{Z} = \text{softmax}\left( \frac{\mathbf{Q} \times \mathbf{K}^\top}{\sqrt{k}} \right) \times \mathbf{V} \in \mathbb{R}^{n \times k}$$
     *注：$\text{softmax}(\mathbf{Q} \times \mathbf{K}^\top / \sqrt{k}) \in \mathbb{R}^{n \times n}$（attention matrix），softmax按行执行*
  3. For $\mathbf{W}_C \in \mathbb{R}^{k \times d}$, calculate $\tilde{\mathbf{U}} = \mathbf{Z} \times \mathbf{W}_C \in \mathbb{R}^{n \times d}$

- The weight matrices are not dependent on the sequence length $n$!!

### Transformer blocks
- Multi-head attention:
  $$\tilde{\mathbf{U}} = \sum_{i=1}^h \text{softmax}\left( \frac{(\mathbf{X}\mathbf{W}_Q^{(i)}) (\mathbf{X}\mathbf{W}_K^{(i)})^\top}{\sqrt{k}} \right) \times (\mathbf{X}\mathbf{W}_V^{(i)}) \times \mathbf{W}_C^{(i)} \in \mathbb{R}^{n \times d}$$

- For each row of $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_n]^\top \in \mathbb{R}^{n \times d}$ and $\tilde{\mathbf{U}} = [\tilde{\mathbf{u}}_1, \dots, \tilde{\mathbf{u}}_n]^\top \in \mathbb{R}^{n \times d}$:
  - Calculate $\mathbf{u}_i = \text{LayerNorm}(\mathbf{x}_i + \tilde{\mathbf{u}}_i; \gamma_1, \beta_1)$ (Add & Normalize)
  - Calculate $\tilde{\mathbf{z}}_i = \mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{u}_i + \mathbf{b}_1) + \mathbf{b}_2$ (Feed Forward), where $\mathbf{W}_1 \in \mathbb{R}^{m \times d}$, $\mathbf{b}_1 \in \mathbb{R}^m$, $\mathbf{W}_2 \in \mathbb{R}^{d \times m}$, $\mathbf{b}_2 \in \mathbb{R}^d$
  - Calculate $\mathbf{z}_i = \text{LayerNorm}(\mathbf{u}_i + \tilde{\mathbf{z}}_i; \gamma_2, \beta_2)$ (Add & Normalize)

- Layer normalization:
  $$\text{LayerNorm}(\mathbf{z}; \gamma, \beta) = \gamma \frac{\mathbf{z} - \mu_{\mathbf{z}}}{\sigma_{\mathbf{z}}} + \beta$$
  - $\mu_{\mathbf{z}} = \sum_{i=1}^d z_i / d$, $\sigma_{\mathbf{z}} = \sqrt{\sum_{i=1}^d (z_i - \mu_{\mathbf{z}})^2 / d}$
  - Normalize activations to have learned mean $\beta$ and variance $\gamma^2$
  - Applied per-sequence (not across batches)


### Self-attention at a high level
- Suppose that the following sentence is an input sentence we want to translate:
  > "The animal didn’t cross the street because it was too tired"
- What does "it" refer to? (street/animal) Simple for humans, not for algorithms
- When processing "it", self-attention lets the model associate "it" with "animal"
- As the model processes each word, self-attention lets it reference other positions for better encoding


### Self-attention at a high level
- Maintaining a hidden state lets RNNs combine previous/current word representations
- Self-attention is how Transformers "bake" understanding of relevant words into the current word’s encoding
  ![figure10](./figure/figure10.png)

  ![figure11](./figure/figure11.png)
- ● The first step in calculating self-attention is to create two vectors from each of the input vectors of the encoder (in this case, the embedding of each word)
- ● For each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process
- ● What are the “query”, “key”, and “value” vectors? They are abstractions that are useful for calculating and thinking about attention


  ![figure12](./figure/figure12.png)
- ● The second step in calculating self-attention is to calculate a score. Say we are calculating the self-attention for the first word in this example, “Thinking”
- ● We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position
- ● The score is calculated by taking the dot product of the query vector with the key vector of the respective word we are scoring


  ![figure13](./figure/figure13.png)
- ● The third and fourth steps are to divide the scores by 8 (the square root of the dimension of the key vectors 64. This leads to having more stable gradients)
- ● Then pass the result through a softmax operation. Softmax normalizes the scores so they are all positive and add up to 1
- ● This softmax score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it is useful to attend to another word that is relevant to the current word

- The fifth step is to multiply each value vector by the softmax score. The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (e.g., by multiplying them by tiny numbers like 0.001)
- The sixth step is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word)


  ![figure14](./figure/figure14.png)
- That concludes the self-attention calculation. The resulting vector is one we can send along to the feed-forward neural network
- In the implementation, this calculation is done in matrix form for faster processing
- The first step is to calculate the Query, Key, and Value matrices, by packing inputs into a matrix $\mathbf{X}$ and multiplying it by the weight matrices we have trained


  ![figure15](./figure/figure15.png)
- Finally, since we are dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer


  ![figure16](./figure/figure16.png)





