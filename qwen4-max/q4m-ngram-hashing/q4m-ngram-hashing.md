# N-gram Hashing

Qwen4-Max adds a hashed N-gram Embedding at its second layer, a large set of memory tables addressed by the short token sequences that end at each position. Qwen has not released Qwen4-Max weights or a technical report, but its announcement names this mechanism for the model, and Qwen's architecture paper builds it on the Engram conditional memory of Cheng et al. (2026). This problem covers the addressing step, which turns every position into one table row per hash head before any vector is read.

## Why n-grams need a hash

Language is full of short phrases that recur, such as names, units and fixed expressions. A direct way to store something about each phrase is a table with one row per n-gram. With a vocabulary of $V$ tokens there are $V^2$ possible bigrams and $V^3$ possible trigrams, so a full n-gram vocabulary for 150,000 tokens would need more than $10^{10}$ rows for bigrams alone, and almost all of them would never be used.

Hashing replaces the full vocabulary with a fixed table of $M$ rows. A hash function sends every n-gram to a row index between $0$ and $M-1$, so the memory cost is set by $M$ instead of $V^n$. The price is collisions, where two different n-grams land on the same row and have to share one vector.

## The suffix n-gram at each position

Engram uses the suffix n-grams that end at position $t$. The bigram at $t$ is $(x_{t-1}, x_t)$, and the trigram adds $x_{t-2}$. A suffix only looks backwards, so the memory never leaks information from later tokens. At the start of a sequence there are not enough earlier tokens, and those slots are filled with a pad id $p$.

Engram also maps raw token ids to canonical ids before hashing. The Qwen paper reports that this token normalization gave no consistent gain, so this exercise hashes raw ids.

## The multiplicative-XOR mix

The Engram paper describes the hash only as a lightweight multiplicative-XOR hash, and the exercise takes the exact formula from the official Engram demo code. Each token is multiplied by an odd multiplier that depends on its offset, and the products are combined with bitwise XOR.

$$
\operatorname{mix}_{t,n}=\bigoplus_{k=0}^{n-1}x_{t-k}\,m_k
$$

Here $m_k$ is the multiplier for offset $k$ and $\oplus$ is XOR on 64-bit integers. Because each offset has its own multiplier, the bigram (6, 9) and the bigram (9, 6) produce different mixes. The mix for order $n+1$ is the mix for order $n$ combined with one more product, so higher orders extend the work of lower ones.

The row for head $j$ of order $n$ is $z_{t,n,j}=\operatorname{mix}_{t,n}\bmod M_{n,j}$. The remainder is taken once, after the XOR, so every bit of the full mix can influence the row.

## Why several prime-sized heads

Any single table of realistic size will see collisions. Engram reduces their cost by giving each order several independent heads, each with its own table, and the retrieved vectors are concatenated later. Two n-grams that collide in one head rarely collide in every head, so the concatenated memory still tells them apart.

The heads differ only in their table sizes, which are distinct primes. Remainders by different primes behave almost independently, so a collision under one prime says little about the others. A size with many small factors, such as a power of two, would keep only the low bits of the mix and discard the rest.

## Why the lookup can be prefetched

Nothing in this computation depends on model weights or hidden states. The rows are a pure function of the token ids, so they are known as soon as the input is tokenized. The Qwen paper places the layer at depth two so that fetching rows from host memory overlaps with the first layer, which is what lets 328 billion N-gram parameters in Qwen4-Max live outside accelerator memory.

## A four-token example

Take the tokens 6, 9, 6, 9 with pad id 1, multipliers 5, 3 and 7, one bigram head of size 13 and one trigram head of size 17.

- **Position 0.** The bigram mixes $30\oplus3=29$, which gives row 3. The trigram adds the pad id times 7, and $29\oplus7=26$ gives row 9.
- **Position 1.** The bigram (6, 9) mixes $45\oplus18=63$, which gives row 11. The padded trigram gives $63\oplus7=56$, which is row 5.
- **Position 2.** The bigram (9, 6) mixes $30\oplus27=5$, which gives row 5 and differs from the row of (6, 9). The trigram gives $5\oplus42=47$, which is row 13.
- **Position 3.** The bigram is (6, 9) again, so it returns to row 11. The trigram gives $63\oplus63=0$, which is row 0.

## Common mistakes to avoid

- **Reading future tokens.** Shifting the sequence the wrong way builds each n-gram from later positions and breaks causality.
- **Padding with zero.** The pad id is a real input, and replacing it with zero moves every row near the sequence start.
- **Adding the products.** Summation is a different hash, so every row stops matching the one the tables were trained with.
- **Reusing one multiplier.** With a shared multiplier the XOR ignores token order, so (6, 9) and (9, 6) collide in every head.
- **Taking the remainder too early.** Reducing each product before the XOR gives different rows than reducing the full mix once.
- **Ordering heads by index first.** All heads of order 2 come before any head of order 3, and this order decides which table each column addresses later.