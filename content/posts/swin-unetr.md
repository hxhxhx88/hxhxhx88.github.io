---
author: "h(x)"
title: "Swin UNETR Note"
date: 2022-09-08
tags: ["Medical Imaging", "Paper Note"]
draft: false
---

[Swin UNETR](https://arxiv.org/abs/2201.01266) is a model using [Swin Transformer](https://arxiv.org/abs/2103.14030) and a U-shaped network architecture to perform medical image segmentation. Its official implementation is available [here](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21). In this memo I write down some key points, without elaborating all accurate details.

![](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/BRATS21/assets/swin_unetr.png?raw=true)

# Input and Output

The input is a $(N, C_i, D, H, W)$ tensor, in which
- $N$ is the batch size.
- $C_i$ is the number of input features.
- $D, H, W$ are the size of the feature, i.e. the medical image.

The output is a $(N, C_o, D, H, W)$ tensor, in which
- $N$ is the batch size.
- $C_o$ is the number of output features, corresponding to the number of segmentation labels.
- $D, H, W$ are the same as input.

(From now on we will ignore the batch dimension.)

Take the [BraTs21](https://www.synapse.org/#!Synapse:syn27046444/wiki/616992) challenge as an example. Each sample consists of four 3D images coming from different [mpMRI sequences](https://en.wikipedia.org/wiki/MRI_sequence) of the same region of interest(RoI):
- T1
- T1ce
- T2
- FLAIR 

![](https://www.researchgate.net/publication/340699988/figure/fig3/AS:881307630444555@1587131521187/Axial-view-of-T1-T1ce-T2-and-Flair.png)

 The annotation is saved in a single 3D image consisting of three non-zero voxel values:
- The necrotic tumor core(NCR 坏死肿瘤细胞) = 1;
- The peritumoral edema(ED 瘤周水肿) = 2;
- The enhancing tumor(ET 增强肿瘤) = 4;

while the expected prediction labels are combinations of the above:
- Tumor core(TC) = NCR + ET;
- Whole tumor(WT) = NCR + ET + ED;
- Enhancing tumor(ET) = ET.

![](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/BRATS21/assets/fig_brats21.png?raw=true)

Therefore, $C_i=4$ and $C_o=3$. In particular, the outputs are probabilities
$$
\mathbb{p}(c,i,j,k)\in[0,1]
$$

telling for the voxel at $(i, j, k)$ the probability for it to be of label TC($c=0$), WT($c=1$) and ET($c=2$).

# Model

Let $D$ be the dimensionality of the problem. For clarity and generality, we set
- $\mathbf{D}=(d_1,d_2,\cdots,d_D)$ be the dimension of the input;
    - $\mathbf{D}=(H,W)$ for $D=2$;
    - $\mathbf{D}=(C,H,W)$ for $D=3$.
- $\mathbf{p}=(p_1,p_2,\cdots,p_D)$ be the patch size along each dimension.

The overall computation flow is as follows:
![](/images/swin-unetr/overall.svg)

- Swin-transform the input $x$ to get $N(=5)$ hidden states with dimensions $$h_n:=\left(2^nC_h, \frac{\mathbf{D}_p}{2^n}\right)$$ where $n=0,1,\cdots,N-1$ and $C_h$ is an internal feature size.
- Encode $h_n$ to $e_n$ of the same dimension.
- Set $e_{N-1}$ to $d_{N-1}$.
- Combine $e_n$ with $d_{n+1}$ to decode to $d_n$ of the same dimension, for $n = N-2, N-1,\cdots 0$.
- Encode $x$ to $e$ of dimension $(C_h,\mathbf{D})$, then combine with $d_0$ to decode to $d$ of the same dimension.
- Project $d$ to the output $y$ of dimension $(C_o,\mathbf{D})$.

In the Swim Transform step, the minimal units are (featurised) patches, rather than original voxels. Those patches are like tokens in natural language. Only in the very last stage are they joined with the original image at voxel granularity.

## Swin Transformer

When doing Swin Transform, we first turn the input into *patches* of size $\mathbf{p}$, then further partition these patches into *windows* of size $\mathbf{w}$, in each of which self-attention is computed. Since calculating attention preserves dimensions, the output is further downsampled to half its spatial dimensions while *double*, not $2^D$ times, the feature dimension.

![](/images/swin-unetr/swin.svg)

### Patch Embedding

Conceptually there are two steps:
- Partition, i.e. reshape, the input into patches of size $\mathbf{p}$ with feature dimension $C_i$.
- Perform $C_h$ linear transformations on each patch to map them to hidden states.

Practically, it is done by a [single convolution](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L461) with `in_channel` be $C_i$, `out_channel` be $C_h$, `kernal_size` be $\mathbf{p}$ and `stride` be $\mathbf{p}$.

### Shifted Window based Self-Attention

Let $\mathbf{w}$ be the window size and $\mathbf{\Delta}$ be the shift distance along each dimension. We require $\Delta_d<w_d$ to ensure overlapping across windows, thus imply communication between them. Typically, we set $\Delta_d=w_d/2$.

Two ingredients are important:
- Since during shifting the window is padded cyclically by rolling patches at tail to head, an *attention mask* is required to tell which patches are in proximity in the original image, thus among which attention can be calculated. A patch should not pay attention to a padded patch moved from far away.
- A *relative position bias* is introduced to put information of the relative position of two patches into attention.

#### Attention Mask

![](/images/swin-unetr/attention_mask.svg)

As shown in the figure, the whole image is shifted by $\mathbf{\Delta}$ and padded cyclicly by patches 2, 5, 6, 7, 8.
```python
x = torch.roll(x, shifts=shifts, dims=dims)
```

Now, in the newly shifted image, patches with same colours are in proximity in the original unshifted image, and should pay attention to only patches of the same colour. (Note that although, say, 0 and 3 are close in the original image, they belongs to different windows in the shifted image, thus are assigned different colours.)

To represent this, we calculate an **attention mask** $m$ which is a $\left(N_w, V_w,V_w\right)$ dimensional tensor, in which $$N_w=\prod\frac{\mathbf{D}_p}{\mathbf{w}}$$ is the total number of windows, and $$V_w=\prod\mathbf{w}$$ is the volumn of each window, i.e. number of patches within each window.

![](/images/swin-unetr/windows.svg)

Conceptually, we need to calculate self-attention within each row of the above figure, but also limit the calculation to each colour block.

The value of $m$ is
$$
m(n,i,j) = \begin{cases}
   0 &\text{if patch $i$ and patch $j$ are relevant in window $n$} \\\\
   -\infty &\text{otherwise}
\end{cases}
$$

This mask is added to the ordinary attention before calculating softmax to effectively turn attentions among irrelevant patches to zero.

[In code](https://github.com/Project-MONAI/MONAI/blob/342f4aa/monai/networks/nets/swin_unetr.py#L758-L795) (also see [here](https://github.com/microsoft/Swin-Transformer/blob/e43ac64/models/swin_transformer.py#L223-L241)), we first calculate a label tensor $l(n,i)$ of dimension $\left(N_w, V_w\right)$ with value in $\mathbb{Z}$ telling the label of patch $i$ in window $n$. Then we set $m(n,i,j)=l(n,i)-l(n,j)$, which can be efficiently calculated through [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html). Finally, we turn all nonzero entry of $m$ to a sufficiently negative large number.
```python
wd, wh, ww = window_size
sd, sh, sw = shift_size
inf = 100.0

d_ranges = (slice(-wd), slice(-wd, -sd), slice(-sd, None))
h_ranges = (slice(-wh), slice(-wh, -sh), slice(-sh, None))
w_ranges = (slice(-ww), slice(-ww, -sw), slice(-sw, None))

# label patches by relevance
l = 0
d, h, w = dims
img_mask = torch.zeros((d, h, w))
for d, h, w in itertools.product(d_ranges, h_ranges, w_ranges):
    img_mask[d, h, w] = l
    l += 1

# turn img_mask into a (N_w, V_w) tensor.
windows = img_mask.view(d // wd, wd, h // wh, wh, w // ww, ww)
windows = windows.permute(1, 3, 5, 2, 4, 6).contiguous().view(-1, wd * wh * ww)

# calculate attention mask
attn_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-inf)).masked_fill(attn_mask == 0, float(0.0))
```

#### Relative Position Bias

Along dimension $i$ there are $2w_i-1$ relative positions, ranging from $-(w_i-1)$ to $w_i-1$, thus in total there are
$$
R := \prod_{i=1}^{D}(2w_i-1)
$$
different possible relative positions in a window.

Given two positions in a window, $\mathbf{p}$ and $\mathbf{q}$, we need to encode their relative position to a number in $\mathbb{Z}_R$. When all $w_i$s are equal to some $w$, a straight-forward way is to view the relative position in a $(2w-1)$-base number system. In general we can do in a similar fashion:

$$
r(\mathbf{p},\mathbf{q}):=\sum_{i=0}^{D-1}\left(p_i-q_i+(w_i-1)\right)\prod_{j=i+1}^{D-1}\cdot(2w_j-1)
$$

where we add $w_i-1$ to $p_i-q_i$ to turn its value range from $[-(w_i-1),w_i)$ to $[0, 2w_i-1)$.

A **relative position bias** $B^{(k)}$ of dimension $\left(h^{(k)},R\right)$ is learned for $k$-th attention calculation, where $h^{(k)}$ is the number of heads of the attention.

The code to construct such tensor can be found [here](https://github.com/microsoft/Swin-Transformer/blob/e43ac64/models/swin_transformer.py#L100-L115) and [here](https://github.com/Project-MONAI/MONAI/blob/342f4aa/monai/networks/nets/swin_unetr.py#L439-L479). Roughly as
```python
wd, wh, ww = window_size

# relative_position_bias_table is a (R, h) tensor storing learnable parameters representing each relative position.
relative_position_bias_table = nn.Parameter(
    torch.zeros(
        (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1),
        num_heads
    ),
)

coords_d = torch.arange(wd)
coords_h = torch.arange(wh)
coords_w = torch.arange(ww)

# coords is a (3, wd, wh, ww) tensor where coords[n][i][j][k] = i, j, k when n = 0, 1, 2.
coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))

# coords_flatten is a (3, V_w) tensor where coords_flatten[n] tells the n-th coordinate of every position in natural encoding from (i, j, k) to the unique index in [0, V_w), n = 0, 1, 2.
coords_flatten = torch.flatten(coords, 1)

# relative_coords is a (3, V_w, V_w) tensor where relative_coords[n][i][j] tells the *difference* of the n-th coordinate between position i and j.
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

# reshape to (V_w, V_w, 3)
relative_coords = relative_coords.permute(1, 2, 0).contiguous()

# calculate the embedding into [0, R).
relative_coords[:, :, 0] += wd - 1
relative_coords[:, :, 1] += wh - 1
relative_coords[:, :, 2] += ww - 1
relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
relative_coords[:, :, 1] *= 2 * ww - 1
relative_position_index = relative_coords.sum(-1)

# relative_position_index is a (V_w, V_w) tensor where x := relative_position_index[i][j] is the embedded relation position in range [0, R).
# relative_position_bias_table[x] then further gives a vector of length h as the bias for each of h headers.
```

#### Attention

The attention distribution is calculated as
$$
P^{(k)}(n,h,\mathbf{p},\mathbf{q})=\text{softmax}\left(\frac{Q^{(k)}\cdot \left(K^{(k)}\right)^\intercal}{\sqrt{C^{(i)}_h/h^{(i)}}}(n, h,\mathbf{p},\mathbf{q}) + B^{(k)}(\cdot,h,r(\mathbf{p},\mathbf{q}))+m(n,\cdot,\mathbf{p},\mathbf{q})\right)
$$
where indexing by $\mathbf{p}$ and $\mathbf{q}$ is understood under their natural mapping to $[0, V_w)$.

Finally, the attention is given by
$$
a^{(k)}=P^{(k)}\cdot V^{(k)}
$$

### Patch Merging

After finishing calculating window attention, which gives a $(N_w, V_w, C_h)$ tensor, we reshape it back to the original image, i.e. a $(C_h,\mathbf{D}_p)$ tensor $z$. Now we want to down sample it to a $(2C_h, \mathbf{D}_p/2)$ tensor.

![](/images/swin-unetr/patch_merging.svg)

As demonstrated in the figure, we split $z$ to $2^D$ smaller tensors of dimension $(C_h, \mathbf{D}_p/2)$ by sampling at step $2$, and then stack them to a $(2^DC_h, \mathbf{D}_p/2)$ tensor. Next, we perform a linear transformation to turn $2^DC_h$ to $2C_h$.

Code can be found [here](https://github.com/Project-MONAI/MONAI/blob/342f4aa/monai/networks/nets/swin_unetr.py#L704-L725) and [here](https://github.com/microsoft/Swin-Transformer/blob/e43ac64/models/swin_transformer.py#L331-L352):
```python
x0 = x[:, 0::2, 0::2, 0::2, :]
x1 = x[:, 1::2, 0::2, 0::2, :]
x2 = x[:, 0::2, 1::2, 0::2, :]
x3 = x[:, 0::2, 0::2, 1::2, :]
x4 = x[:, 1::2, 0::2, 1::2, :]
x5 = x[:, 1::2, 1::2, 0::2, :]
x6 = x[:, 0::2, 1::2, 1::2, :]
x7 = x[:, 1::2, 1::2, 1::2, :]
x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
x = nn.Linear(8 * dim, 2 * dim, bias=False)(reduction(x))
```

## Encoder

The encoder is a typical multi-layer residual convolutional neural network, with carefully setting `kernal_size` to `3`, `stride` to `1` and `padding` to `1`, thus preserving the dimensions.

Note that in general the output dimension of a convolution operator is given by:
$$
o=\left\lfloor\frac{i+2p-k}{s}\right\rfloor+1
$$
where
- $i$: input dimension;
- $p$: padding on both side;
- $k$: kernel size;
- $s$: stride.

Thus indeed when $s=p=1$ and $k=3$ we have $o=i$.

## Decoder

When decoding, we first use a [transposed convolutional layer](https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11) to up sample the previously decoded tensor $d$ to double its dimensionality from $(C,\mathbf{D})$ to $(2C,2\mathbf{D})$. In general, the dimension of the output of a transposed convolutional is given by:
$$
o = (i-1)\cdot s + k-2p
$$
Thus by setting $k=s=2$ and $p=0$, we have $o=2i$.

After that, we concatenate it with the encoded tensor at the same level to get a $(4C,2\mathbf{D})$ tensor, and put it through another convolutional neural network to reduce $4C$ to $2C$ again.

# Remark

- As in the overall diagram, in order to be able to combine $e$ and $d_0$ to decode to $d$, it requires $\mathbf{D}$ is twice as much as $\mathbf{D}_p$, which further requires $\mathbf{p}$ to be $\mathbf{2}$. A relaxation maybe desired.
- Since shift size is [alternatingly half the window size](https://github.com/microsoft/Swin-Transformer/blob/e43ac64/models/swin_transformer.py#L400) and [accumulated](https://github.com/microsoft/Swin-Transformer/blob/e43ac64/models/swin_transformer.py#L420), it seems crucial to pick *odd* window size, otherwise on the, say, 3rd shift simply a whole window is shifted and there is no information exchange between windows.
