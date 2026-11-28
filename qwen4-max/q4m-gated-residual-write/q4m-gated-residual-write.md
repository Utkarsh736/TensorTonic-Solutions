# Gated Residual Write

The gated residual write is the second half of Qwen4-Max's Gated Residual. Qwen4-Max is a closed model, and Qwen names Gated Residual with four branches and separate read and write gates in its Qwen4-Max announcement. The equations come from Qwen's architecture paper, On the Design of Qwen3.8-Next Architecture (Qwen Team, 2026), Section 2.2, Equations 33 and 34. After a sublayer turns the gated read into an output $y$, the write adds $y$ to every branch with a scale that each branch sets for itself.

## A plain residual adds everything at full strength

In a standard transformer, each sublayer output is added once to the single residual stream, $h \leftarrow h + F(x)$. Every sublayer contributes with weight one, so the stream after $L$ sublayers is the embedding plus the plain sum of all outputs. If one sublayer's output is unhelpful for some later computation, the only way to suppress it is for that sublayer to learn to output less for everyone.

## One output and a scaled write per branch

With $n_r$ branches the write can do better. The scales $s = 2\,\sigma(\frac{1}{n_r} W_w \operatorname{vec}(\widehat R))$ form one number per branch, and branch $i$ receives $R_i + s_i y$. The scales depend on the current position, because they are computed from the normalized branches $\widehat R_i$ there, the same normalized branches the read gate saw.

The write never moves information from one branch to another. Every branch gets the same vector $y$, so the branches differ only in how much of each sublayer they keep. After many layers, branch $i$ holds its starting value plus $\sum_\ell s_i^{(\ell)} y^{(\ell)}$, a weighted sum of all sublayer outputs with its own weights $s_i^{(\ell)}$, where $\ell$ indexes the sublayers. The read then picks per channel which of these differently weighted histories to use.

The Qwen paper reports that making the write elementwise, with one scale per branch and channel, gave almost no gain over one scalar per branch, while refining the read in the same way did help. That is why the write stays a scalar per branch and the read is elementwise.

## Why the scale lies between zero and two

The factor two in front of the sigmoid places every scale in $(0, 2)$. At a pre-activation of zero the scale is exactly one, so a write projection near zero starts each branch as a plain residual addition. The paper states the formula without giving a reason for the bound. One reading is that the bound lets a branch nearly skip a sublayer with a scale near zero, or double it with a scale near two, while preventing any single write from amplifying an output without limit.

## Conventions this exercise fixes

The paper adopts zero-centered RMSNorm without printing it, so this exercise uses the Qwen3-Next definition, $x / \sqrt{\operatorname{mean}(x^2) + 10^{-6}} \cdot (1 + \gamma)$. The vec in Equation 33 uses branch-major order, so branch $i$ occupies positions $id$ to $id + d - 1$. The helper in the starter code computes the normalized branches, and the scales use those normalized branches while the update adds $s_i y$ to the raw branches $R_i$.

## Two branches with a neutral and an active row

Take $n_r = 2$, $d = 1$, branches $R_1 = 2$ and $R_2 = -1$, norm weights of zero, and sublayer output $y = 0.5$.

- Normalization with width one keeps only the sign, so the stack is $(1.0000, -1.0000)$.
- Write row $(1, 1)$ gives $(1 - 1)/2 = 0$, so $s_1 = 2\,\sigma(0) = 1.0000$ and $R'_1 = 2 + 0.5 = 2.5000$.
- Write row $(3, -1)$ gives $(3 + 1)/2 = 2$, so $s_2 = 2\,\sigma(2) = 1.7616$ and $R'_2 = -1 + 1.7616 \cdot 0.5 = -0.1192$.

The two branches received the same output and kept different amounts of it, and neither branch's value leaked into the other.

## Common mistakes to avoid

- **Computing scales from raw branches.** The scales come from the normalized branches, so raw values make them depend on branch magnitude.
- **Dropping the factor two.** A plain sigmoid caps every scale at one and gives one half at a zero pre-activation, which halves every residual write.
- **Adding to the normalized branches.** The output is added to the raw branches, and writing into the normalized ones throws away the residual magnitude.
- **Sharing one scale across branches.** Averaging or reusing one scale removes the only thing that makes the branches differ.
- **Forgetting the one over n factor.** Without it the sigmoid input grows with the number of branches and the scales saturate toward zero or two.
- **Normalizing the sublayer output.** The write adds the output exactly as the sublayer returned it, and normalizing it first changes how much every branch receives.