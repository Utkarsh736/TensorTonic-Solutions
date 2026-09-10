# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Utkarsh Tomar's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/utkarshtomar736.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Data Augmentation | Implement AlexNet image augmentation operations for deterministic crops, horizontal flips, and intensity changes. | https://www.tensortonic.com/research/alexnet/alexnet-augmentation |
| AlexNet Convolution Layer | Implement an AlexNet convolutional layer with learned filters, bias, stride, padding, and multi-channel outputs. | https://www.tensortonic.com/research/alexnet/alexnet-conv-layers |
| Dropout Regularization | Implement inverted dropout for AlexNet with seeded masks and training-versus-inference behavior. | https://www.tensortonic.com/research/alexnet/alexnet-dropout |
| Local Response Normalization | Implement AlexNet local response normalization across neighboring channels using the paper's scaling equation. | https://www.tensortonic.com/research/alexnet/alexnet-lrn |
| Overlapping Max Pooling | Implement AlexNet overlapping max pooling with a 3x3 window and stride two across spatial dimensions. | https://www.tensortonic.com/research/alexnet/alexnet-pooling |
| ReLU Activation Function | Implement AlexNet's elementwise ReLU activation, preserving positive values while setting negative values to zero. | https://www.tensortonic.com/research/alexnet/alexnet-relu |
| WordPiece Tokenization | Implement BERT WordPiece tokenization with greedy longest-match subwords, continuation prefixes, and unknown-token fallback. | https://www.tensortonic.com/research/bert/bert-wordpiece |
| GAN Generator | Implement a GAN generator that transforms latent noise through learned dense layers into generated samples. | https://www.tensortonic.com/research/gan/gan-generator |
| Bottleneck Block | Build a ResNet bottleneck block using 1x1 channel reduction, 3x3 convolution, and 1x1 channel expansion. | https://www.tensortonic.com/research/resnet/resnet-bottleneck |
| Convolutional Block | Implement a ResNet convolutional block with a projected shortcut that matches changed spatial and channel dimensions. | https://www.tensortonic.com/research/resnet/resnet-conv-block |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| VAE Encoder | Implement a variational autoencoder encoder that maps inputs to separate latent mean and log-variance vectors. | https://www.tensortonic.com/research/vae/vae-encoder |
| Class Token [CLS] | Prepend a learned classification token to each Vision Transformer patch sequence for image-level prediction. | https://www.tensortonic.com/research/vit/vit-class-token |
| ViT Encoder Block | Build a Vision Transformer encoder block with layer normalization, multi-head attention, MLP, and residual connections. | https://www.tensortonic.com/research/vit/vit-encoder-block |
| Classification Head | Implement the Vision Transformer classification head by normalizing and projecting the final class-token representation. | https://www.tensortonic.com/research/vit/vit-mlp-head |
| Patch Embedding | Implement Vision Transformer patch embeddings by splitting images into fixed patches and linearly projecting each patch. | https://www.tensortonic.com/research/vit/vit-patch-embedding |
| Position Embedding | Add learned positional embeddings to Vision Transformer patch-token sequences while preserving batch dimensions. | https://www.tensortonic.com/research/vit/vit-position-embedding |
| GELU | Implement exact GELU activation in CUDA with one thread per element and the device error-function intrinsic. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/gelu |
| Leaky ReLU | Implement Leaky ReLU activation in CUDA with one thread per element, bounds checks, and a configurable negative slope. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/leaky-relu |
| Matrix Addition | Implement elementwise matrix addition in CUDA with a two-dimensional grid, row-major indexing, and bounds checks. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/matrix-addition |
| ReLU | Implement ReLU activation in CUDA with one thread per element, bounds checks, and branch-efficient rectification. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/relu |
| Sigmoid | Implement sigmoid activation in CUDA with one thread per element, device exponential math, and bounds-checked memory access. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/sigmoid |
| Swish | Implement fused Swish or SiLU activation in CUDA with one thread per element and device exponential math. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/swish |
| Tanh | Implement hyperbolic tangent activation in CUDA with one thread per element, device intrinsic math, and bounds checks. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/tanh |
| Vector Addition | Implement bounds-checked pointwise vector addition in CUDA with one thread per output element. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/vector-addition |
| Vector Subtraction | Implement bounds-checked pointwise vector subtraction in CUDA with one thread per output element. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/vector-subtract |
| Fused Multiply-Add | Implement a Triton fused multiply-add kernel with contiguous tiles, hardware FMA, and masked tail handling. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-fused-multiply-add |
| GELU | Implement exact GELU activation in Triton with device error-function math and masked contiguous tiles. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-gelu |
| Vector Max Reduction | Compute a vector maximum with one Triton reduction program and masked tail lanes that cannot win comparisons. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-max |
| ReLU | Implement ReLU activation in Triton with contiguous program tiles, branch-free rectification, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-relu |
| SiLU | Implement fused SiLU or Swish activation in Triton with contiguous tiles, sigmoid weighting, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-silu |
| Vector Sum Reduction | Implement tiled vector sum reduction in Triton with register partials, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-sum |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-vector-addition |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/utkarshtomar736)
<!-- tensortonic:end -->
