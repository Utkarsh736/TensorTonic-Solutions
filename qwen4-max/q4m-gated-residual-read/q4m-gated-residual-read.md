# Gated Residual Read

The gated residual read is how a Qwen4-Max sublayer gets its input from a residual stream that has been widened into four branches. Qwen4-Max is a closed model, and Qwen names Gated Residual with four branches and separate read and write gates in its Qwen4-Max announcement. The equations come from Qwen's architecture paper, On the Design of Qwen3.8-Next Architecture (Qwen Team, 2026), Section 2.2, Equations 30 to 32. This problem covers the read, which normalizes every branch, gates every channel of every branch, and averages the result.

## One stream and a pre-norm

A standard pre-norm transformer keeps one residual vector $h \in \mathbb{R}^{d}$ per token. Each sublayer $F$ reads $x = \operatorname{RMSNorm}(h)$ and adds its output back, $h \leftarrow h + F(x)$. Every sublayer sees the same single stream, so whatever earlier layers wrote is mixed together before any later layer can look at it.

Widening the stream to $n_r$ branches $R_1, \dots, R_{n_r}$ gives the network several places to keep information apart, while a sublayer still expects one vector of width $d$. The read is the operator that turns the branches into that one input.

## Four branches and an elementwise read

The read first normalizes each branch with its own weight $\gamma_i$, which gives $\widehat R_i$. Because each branch is normalized separately, a branch with a large norm cannot drown out the others, and the read already delivers a normalized vector to the sublayer. This is why Gated Residual removes the pre-norm in front of each block.

The gate looks at every normalized branch at once. The stacked vector of length $n_r d$ passes through a narrow bottleneck of rank $k$, a SiLU, a projection back to length $n_r d$, and a sigmoid. The result $G$ has one value in $(0, 1)$ for every channel of every branch, so the sublayer can take channel 3 mostly from one branch and channel 5 mostly from another. The factor $1/n_r$ inside the SiLU keeps the bottleneck input on the scale of a single branch, and the final average keeps the read on the scale of one normalized vector.

## Dropping the mixing matrix of hyper-connections

Hyper-connections (Zhu et al., 2025) also widen the residual stream into branches. They read with one scalar weight per branch, write with one scalar per branch, and add an $n_r \times n_r$ matrix that mixes the branches with each other at every block, which the manifold-constrained variant mHC forces to be doubly stochastic. The Qwen paper reports that once the read is refined to one weight per branch and channel, the mixing matrix adds no significant gain, so Gated Residual drops it. Without it a branch changes only through sublayer writes, each block avoids an extra pass over the whole residual state, and the constraints mHC needs are no longer required.

Kimi K3's Attention Residuals solve a related problem differently, since they attend over the outputs of earlier layers, whereas Gated Residual keeps four parallel branches and gates its read per element and per branch.

## Conventions this exercise fixes

The paper adopts zero-centered RMSNorm without printing it, so this exercise uses the Qwen3-Next definition, $x / \sqrt{\operatorname{mean}(x^2) + 10^{-6}} \cdot (1 + \gamma)$, where a weight of zero means a scale of one. Equation 31 stacks the branches with vec, and this exercise fixes branch-major order, so branch $i$ occupies positions $id$ to $id + d - 1$. The factor $1/n_r$ sits inside the SiLU, exactly where Equation 31 prints it.

## Two branches of width one

Take $n_r = 2$, $d = 1$, branches $R_1 = 3$ and $R_2 = -0.5$, norm weights $\gamma = (0, 0.2)$, rank $k = 1$, $W_d = (1, 1)$ and $W_u = (2, -2)^{\mathsf T}$.

- With width one, normalization keeps only the sign and the scale, so $\widehat R_1 = 1.0000$ and $\widehat R_2 = -1.2000$. The raw size difference between 3 and 0.5 is gone.
- The bottleneck input is $(1.0000 - 1.2000) / 2 = -0.1000$, and $\operatorname{SiLU}(-0.1000) = -0.0475$.
- The gates are $\sigma(-0.0950) = 0.4763$ and $\sigma(0.0950) = 0.5237$.
- The read is $(0.4763 \cdot 1.0000 + 0.5237 \cdot (-1.2000)) / 2 = -0.0761$.

The branch that was six times larger in raw value contributes slightly less than the small one, because the read compares branches only after normalization.

## Common mistakes to avoid

- **Sharing one norm weight.** Every branch has its own row of the norm weight, and using the first row for all of them erases what each branch learned to emphasize.
- **Using standard RMSNorm.** The stored weight is an offset from one, so multiplying by the weight instead of one plus the weight shrinks a zero-initialized norm to zero.
- **Moving the one over n factor.** Dividing after the SiLU changes the bottleneck activation, since SiLU is not linear.
- **Replacing the outer sigmoid.** A SiLU or no activation lets gates leave the range zero to one and even turn negative.
- **Summing instead of averaging.** Without the final division by the number of branches the read grows with the branch count.
- **Gating the raw branches.** The read multiplies the gates with the normalized branches, and raw branches bring back the scale differences the norm removed.
- **Interleaving the stack.** Stacking channel-major sends each projection weight to the wrong branch and channel.