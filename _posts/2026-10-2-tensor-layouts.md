---
layout: post
title: "Visualizing N-Dimensional Tensor Layouts: Reshape and Permute"
tags: [Deep Learning, LLMs, PyTorch]
folder: tensor-layouts
pinned: false
toc: true
---

<!-- Only needed if the post uses math. -->
{% include mathjax.html %}

---

Most tensor-based code relies on `reshape`, `transpose` or `permute` operations. Some patterns are easy to recognize and understand without having to go into memory layout, strides, view, and contiguity territory. However, when dealing with high-dimensional arrays, these ops are easy to get *silently* wrong without some kind of mental model and quick intuition about what is going on under the hood.

The goal of this post is to build some **layout intuition** for these operations: how tensors are structured internally, why they sometimes need to be permuted or why some calls require a copy of the data and some can get away with a view of the original tensor. We will start with the basics and then we will see Grouped-Query Attention as an example.

> Some figures are a tad too big, so probably the phone experience is not the best. 

> Claude has very politely made all the figures interactive so that you can hover over any cell to follow the element through the panels in the figure.

You can click [here](#grouped-query-attention) if you want to skip to the GQA part.

---

# Tensor internals

A tensor in PyTorch (or NumPy) is not stored as a grid, it's a single flat strip of numbers plus some metadata. We mostly care about two fields in this metadata: the **shape** of the tensor and its **stride**. The shape tells us how many elements we have in each dimension, and the stride tells us how many elements we have to skip in memory to increment that axis index by one.

Let's get a sample tensor `x = torch.arange(40).reshape(4, 2, 5)` and say that it represents a `(batch, height, width)` tensor. The height corresponds to the number of rows, and the width corresponds to the number of columns. The convention that PyTorch follows is **row-major order**, which means that the last index changes the fastest. 

As you can see in the figure below, the reshape puts the first element in the first column, and then places the second one in the second column. This is what it means to "change the fastest". Only when all the columns in that row have been filled, we increment the height index and continue with the columns in the next row. Once both rows are filled, we continue with the next batch element.

<figure class="tl-fig" id="fig-strip">
  <div class="tl-stage"></div>
  <p class="tl-readout"><code class="language-plaintext highlighter-rouge">x[2, 1, 3]</code> is at address <code class="language-plaintext highlighter-rouge">2·10 + 1·5 + 3·1 = 28</code>. Hover a cell to try another one.</p>
</figure>

This links to the concept of strides we have briefly mentioned earlier. **For a contiguous tensor** (I'm getting a bit ahead of myself here), each stride is the product of the sizes of the dimensions to its right. As an example, for `x` with shape `(4, 2, 5)`, the strides are `(10, 5, 1)`. As mentioned above, the strides tell us how many elements we have to "skip" in order to increase that axis index by one. In the case of our batch dimension, we need to skip 10 elements. The caption of the figure shows this computation for every element in the tensor. Mathematically:

$$ \text{address} = i_0 \cdot s_0 + i_1 \cdot s_1 + i_2 \cdot s_2 \quad ; \quad s_k = \prod_{j > k} n_j $$

This formula is the mental model we will try to internalize, visualizing how `reshape` and `permute` change either how we group the elements or the strides of each axis.

# Reshaping or Viewing a tensor

When we `reshape`/`view` a tensor we are not doing anything other than taking the internal layout of the tensor and defining how many dimensions we want and how many elements we want in each dimension. We are regrouping the strip.

> `view` and `reshape` are similar. The difference is that if the tensor has incompatible strides, `reshape` will create a contiguous copy before regrouping, while `view` will raise an error. 

<figure class="tl-fig" id="fig-reshape">
  <div class="tl-controls" role="group" aria-label="Shape">
    <button data-k="shape" data-v="40" aria-pressed="false">(40,)</button>
    <button data-k="shape" data-v="4,10" aria-pressed="false">(4, 10)</button>
    <button data-k="shape" data-v="4,2,5" aria-pressed="true">(4, 2, 5)</button>
    <button data-k="shape" data-v="8,5" aria-pressed="false">(8, 5)</button>
    <button data-k="shape" data-v="5,8" aria-pressed="false">(5, 8)</button>
  </div>
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

# Permuting or Transposing a tensor

`permute` changes the order of the axes. Let's look at what `y = x.permute(1, 0, 2)` is doing:
- Axis `1` of `x` becomes axis `0` of `y`.
- Axis `0` of `x` becomes axis `1` of `y`.
- Leaves axis `2` in the same place. 

> `transpose` is just a shortcut to swap two axes. Considering the previous operation: `x.permute(1, 0, 2) == x.transpose(0, 1) == x.transpose(1, 0)`.

## Two dimensions: easy peasy, we saw this in school

Permuting two axes is what most people understand by the "transpose" of a matrix. If we imagine a 2D matrix, transposing it means to mirror it across the diagonal, swapping the `(i, j)` element with the `(j, i)` element. 

In the example below, we see that we are swapping the dims, but **also the strides**. The addresses in memory don't change, only their arrangement in the tensor and how we read them. A very important point: you might have noticed that by swapping the strides together with the dimensions, we have broken the rule that defines a contiguous tensor, the strides are no longer the product of the dimensions to the right, so we **no longer have a contiguous tensor**.

<figure class="tl-fig" id="fig-mirror">
  <div class="tl-stage"></div>
  <p class="tl-readout">Hover a cell: it keeps its address and lands on the other side of the dashed diagonal.</p>
</figure>

## Three dimensions: a bit less easy

We could visualize this as a 3D cube in space, but I want to use this "easier" case as a way to start visualizing N dimensions in a plane. If we have three dimensions, we can keep the last two as rows and columns, just like before, and place the "batch elements" (each index of the first axis) next to each other.

The following figure shows all the possible permutations in a three-axis tensor. Pay special attention to how the elements of the output tensor change when we swap the axes. The shape is "easy" to predict, but the order of the elements is a bit less intuitive. 

<figure class="tl-fig" id="fig-perm3">
  <div class="tl-controls" role="group" aria-label="Permutation">
    <button data-p="012" aria-pressed="true">(0, 1, 2)</button>
    <button data-p="102" aria-pressed="false">(1, 0, 2)</button>
    <button data-p="021" aria-pressed="false">(0, 2, 1)</button>
    <button data-p="210" aria-pressed="false">(2, 1, 0)</button>
    <button data-p="120" aria-pressed="false">(1, 2, 0)</button>
    <button data-p="201" aria-pressed="false">(2, 0, 1)</button>
    <button data-replay>↻ replay</button>
  </div>
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

`(0, 1, 2)` is trivial, `(0, 2, 1)` is just a 2D transpose in each batch element. 

`(1, 0, 2)` and `(2, 1, 0)` also swap two axes, we can still use the "mirroring" visualization, but this time around elements that are not visually together in our mental layout. Instead of trying to mirror the tensor, let's try to use these as a stepping stone to more complex permutes.

For the `(1, 0, 2)` case: the output shape will be `(2, 4, 5)`. We can tell by using the strides that all the blue elements will fall in a batch item and all the orange ones will fall in the other one. Another way to arrive at the same layout is to see that *we will have two batch elements instead of four, and four rows instead of two, so we need to "pick" the row with the same index from every batch element and fill each new batch element with all of them*, **all items sharing a row index in `x` will now share a batch index in `y`**. `(2, 1, 0)` is also a two-axis swap that involves the batch axis, you can try to predict the output layout.

Both `(1, 2, 0)` and `(2, 0, 1)` permute all three axes at the same time. Let's leave the stride arithmetic aside and use the same logic to get the layout "visually".

In the `(1, 2, 0)` case: we have two batch items now, so we know each one will group the rows that shared the same row index in `x`; we now have five rows per batch element in `y`, so each row will correspond to items that had the same column index in `x`; and finally we have four columns, where each column index now corresponds to the items that shared the same batch index before applying `permute`. The reasoning for `(2, 0, 1)` is left as an exercise for the reader.

## More dimensions: spooky

For a larger number of dimensions I find it easier to try and visualize each dimension as the magnitude it denotes. Doing this makes the "row-major" intuition a bit brittle, so we have to be careful to walk the axes from right to left when flattening the memory layout to, for example, reshape the tensor.

The figures below show an image being cut into patches for a ViT (without using a Conv2D op). If we take a `C, H, W = (3, 6, 6)` image and cut it into `3x3` patches, we should end up with 4 patches of $$ 3 \times 3 \times 3 = 27 $$ elements.

```python
img.shape
# (3, 6, 6): channel, row, column
cut = img.reshape(3, 2, 3, 2, 3)
# channel, patch row, pixel row, patch col, pixel col
moved = cut.permute(1, 3, 0, 2, 4)
# patch row, patch col, channel, pixel row, pixel col
patches = moved.reshape(4, 27)
# patch, value
```

We start with the image as three blocks, one per channel:

<figure class="tl-fig" id="fig-patch-0">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

We `reshape` the image from `(channels, height, width)` into `(channels, patches_row, patch_height, patches_col, patch_width)`. The arrows below show what I meant by visualizing each dimension as its magnitude. Take into account that you are now working with a 5-D tensor (!).

<figure class="tl-fig" id="fig-patch-1">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

<!-- TODO: I think this is a bit messy explanation-wise, not a fan of the rightmost stuff etc. -->

As we saw before, the memory remains untouched, we have just recomputed the strides and changed the shape of the tensor. Now let's look at where each dimension goes when we do `cut.permute(1, 3, 0, 2, 4)`, we have to keep the arrows and axis numbers that hold the "structure" of the dimensions the same, but we can swap the dimensions freely. The right-most dimension remains the same, the patch width; the second dimension from the right is now the patch height, so we lay the whole $$3 \times 3$$ patch side by side. The third dimension from the right is now the channels dimension, so we stack each flattened patch channel on top of each other. The two dimensions left are the number of patches along both the height and the width of the original image, if we "abstract" the other three dimensions and focus on these two, we can imagine that we are looking at a $$2 \times 2$$ tensor where each element is a whole patch.

<figure class="tl-fig" id="fig-patch-2">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

For completeness, let's apply the final reshape we would perform before the linear projection. Since we have scrambled the axes and made the tensor non-contiguous, PyTorch performs a copy of the data and reorders its internal representation before reshaping (see next section for more info on this operation). We reshape to `(patches_row * patches_col, channels * patch_height * patch_width) == (4, 27)`. We have to be careful here with the mental image, we have laid out the axes in a way that was visually convenient for us in "conceptual" terms for the patches, but when flattening the data to reorganize it the reshape follows row-major ordering. Make sure that the axes that are being merged are "neighbors" and go from slowest to fastest.

<figure class="tl-fig" id="fig-patch-3">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

# Making a tensor contiguous 

In all the `permute` figures we have kept the memory strip at the bottom the same. As we have already mentioned a couple of times, `permute` doesn't change the numbers at all, it just shuffles the shapes and the corresponding strides. The formula for the memory address still works, we just change how we walk over the strip, essentially jumping over different "sections" of it. Going back to our simpler example, let's take a look at what this walking looks like:

<figure class="tl-fig" id="fig-read">
  <div class="tl-stage"></div>
  <p class="tl-readout">Each solid line is a run of five cells read in one go. At the dot the reading jumps, and the arrowhead lands on the cell where it continues.</p>
</figure>

<!-- TODO: Rephrase the paragraph below to reflect that is either one of the two and that a view of a non contiguous tensor is legal in some cases -->

Once again, this is what PyTorch calls a **non-contiguous** tensor: the memory order and the layout have come apart. As we have seen, `reshape` will work fine with a non-contiguous tensor, while `view` will raise an error. This is because `reshape` creates a copy of the tensor and reorders the underlying data into a contiguous tensor if needed. To be more precise, a `view` will only work if:
- Each new dimension is a subspace of an original dimension (e.g. `unflatten`ing a dimension).
- The reshape covers a set of original dimensions $$ d, d+1, ..., d+k $$ that, for all $$ i = d, ..., d+k-1, $$ satisfy $$ \text{stride}[i] = \text{stride}[i+1] \times \text{shape}[i+1] $$.

Under the hood, `reshape` performs (if needed) a `torch.Tensor.contiguous()` which basically returns itself if already contiguous, or allocates a copy and writes the data in contiguous order. This is shown in the figure below.

<figure class="tl-fig" id="fig-contig">
  <div class="tl-stage"></div>
  <p class="tl-readout">Hover a ribbon to follow one run of five numbers from the old strip to the new one.</p>
</figure>

This is the only operation in this post that moves the underlying data (copies, actually).

# Grouped-Query Attention 

As promised, let's go over a more practical example now. Regular multi-head attention has the same number of heads for the queries, keys and values (`n_heads`). [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) keeps `n_heads` for the queries but gives a smaller number of heads to the keys and values (`n_kv_heads`), with each key/value head being shared by `q_per_kv_head = n_heads / n_kv_heads` query heads. The main advantage of this is shrinking the KV cache at inference time.

## Setup

For the example I'll use small numbers to excuse the reader from needing to buy an ultrawide monitor just to read the post. The dimensions and colors for each axis are as follows:

| Name | Value | Meaning |
| ---- | ----- | ------- |
| `B` | Any | Batch (not included in the figures, arbitrary) |
| `T` | 4 | Tokens |
| `n_heads` | 6 | Query heads |
| `n_kv_heads` | 2 | Key/value heads |
| `q_per_kv_head` | 3 | Query heads per key/value head (`n_heads / n_kv_heads`) |
| `d_head` | 5 | Features per head |

<div class="tl-fig tl-legend" id="tl-legend"></div>

The six query heads get six hues: heads 0 to 2 are the cool ones and share the blue key head, heads 3 to 5 are the warm ones and share the orange one.

> I will leave the batch dimension out of the figures but keep it in the shapes and the code.

For reference, the code (ignoring causal masking) should be something along these lines:

```python
# x: (B, T, d_model)
q_per_kv_head = n_heads // n_kv_heads  # 3

q = wq(x)  # (B, 4, 30)
k = wk(x)  # (B, 4, 10)
v = wv(x)  # (B, 4, 10)

q = q.reshape(B, T, n_kv_heads, q_per_kv_head, d_head)
q = q.permute(0, 2, 3, 1, 4)  # (B, 2, 3, 4, 5)
k = k.reshape(B, T, n_kv_heads, 1, d_head)
k = k.permute(0, 2, 3, 1, 4)  # (B, 2, 1, 4, 5)
v = v.reshape(B, T, n_kv_heads, 1, d_head)
v = v.permute(0, 2, 3, 1, 4)  # (B, 2, 1, 4, 5)

att = q @ k.transpose(-2, -1) / d_head ** 0.5  # (B, 2, 3, 4, 4)
att = att.softmax(dim=-1)
out = att @ v  # (B, 2, 3, 4, 5)

out = out.permute(0, 3, 1, 2, 4)  # (B, 4, 2, 3, 5)
out = out.reshape(B, T, n_heads * d_head)  # (B, 4, 30)
```

## Projecting the inputs

The three projections give us one row per token: 30 values for the queries (6 heads of 5 features) and 10 for the keys and the values (2 heads of 5 features). Values look exactly like the keys, so I'm only drawing the keys.

<figure class="tl-fig" id="fig-gqa-0">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

## Splitting into heads

The reshape splits those 30 values into three axes: which key/value head the query head belongs to, which of the query heads of that group it is, and the feature. Going back to our "placing the axes where they make sense spatially", here we keep the tokens going down and lay the three new axes across, from the biggest steps to the smallest: the two groups side by side (one per key/value head), the three query heads inside each group, and the five features inside each head. The keys get the same treatment, with a 1 where the queries have `q_per_kv_head`.

<figure class="tl-fig" id="fig-gqa-1">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

As we already know, it's a reshape, so nothing moved, the rows of the previous figure just got regrouped.

The order of the two new axes is a decision: `(n_kv_heads, q_per_kv_head)` puts query heads 0, 1 and 2 with key/value head 0 and heads 3, 4 and 5 with key/value head 1. Writing `(q_per_kv_head, n_kv_heads)` would pair heads 0, 2 and 4 with key/value head 0 (both are valid architectures: interleaved vs tiled, be careful if loading weights).

## Preparing for the matmuls

`permute(0, 2, 3, 1, 4)` keeps the batch at the front and takes the token axis all the way to the back, next to the features. As we will see in the next subsection, this allows for batched matrix multiplication of queries and keys.

The drawing doesn't need to change at all, this time we change the numbers in the arrows (the axis index of each magnitude). As we have seen in the first half of the post, this matters for reshaping, but we can lay them down visually however we want.

<figure class="tl-fig" id="fig-gqa-2">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

Now, the last two dimensions are (token, feature), so every head is a small matrix with one row per token and one column per feature, and the two axes in front only say which matrix we are looking at: which group, and which head inside the group. The keys are two matrices, one per group.

## Computing the scores and broadcasting

`@` only looks at the last two axes and treats everything in front as a batch, so it works matrix by matrix: the $$ 4 \times 5 $$ matrix of each query head gets multiplied with the matrix of its key head. For the shapes to fit, the keys need their last two axes swapped. `transpose(-2, -1)` is the mirror we have seen previously applied inside every matrix, which leaves them with one row per feature and one column per token.

Axis 2 has size 3 in the queries and size 1 in the keys, so the single key matrix of each group is read once for each of the three query heads of that group (broadcasting). In the figure those extra readings are the dotted copies.

<figure class="tl-fig" id="fig-gqa-3">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

After the softmax, `att @ v` broadcasts in the same way and returns a tensor shaped like the queries, `(B, 2, 3, 4, 5)`, i.e. `(B, n_kv_heads, q_per_kv_head, T, d_head)`.

## Merging the heads back

The output has to go back to one row of 30 values per token. `permute(0, 3, 1, 2, 4)` undoes the first `permute` and brings the tokens back to axis 1, leaving the three axes we want to merge at the end: key/value head, query head inside the group, feature (in order).

<figure class="tl-fig" id="fig-gqa-4">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

Once again the picture is the same and only the numbers on the arrows moved to the order we had right after splitting the heads. Now the last `reshape` can merge the three axes at the end (closing the horizontal gaps in the figure).

Since the permuted tensor is not contiguous, that `reshape` has to make a contiguous copy. The figure below illustrates the `contiguous()` we have already seen for our particular case: every set of five numbers (`d_head`) moves from its place in the old strip, where the memory goes head after head, to its place in the new one, where it goes token after token.

<figure class="tl-fig" id="fig-gqa-5">
  <div class="tl-stage"></div>
</figure>

<figure class="tl-fig" id="fig-gqa-6">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

And we have our `(B, T, n_heads * d_head)` back!

Just for reference, compare this to doing the `reshape` without the `permute`, which gives a valid shape but messes up the structure inside the tensor:

<figure class="tl-fig" id="fig-gqa-7">
  <div class="tl-stage"></div>
  <p class="tl-readout"></p>
</figure>

In this case for example, the first row holds head 0 for all four tokens followed by head 1 for tokens 0 and 1.

---

# Summary

- A tensor is **a 1D array plus a header**. The header holds the shape and the strides, and the address of an element is $$ \sum{\text{index}_i \cdot \text{stride}_i} $$.
- `reshape` and `view` **regroup elements**. They merge or split axes that are already neighbors and don't change the memory order unless `reshape` performs a `contiguous` under the hood.
- `permute` and `transpose` **reorder axes**. They shuffle shape and strides together but don't move anything.
- `contiguous` **moves numbers**. We need it between a permute and a `view`, `reshape` does it when needed.
- With many axes forget about rows and columns and **name every axis by what it measures**, try to give it meaning spatially and check their order before reshaping.

---

Thanks for reading!

---

# Related links

- [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/) (the part about strides)
- [torch.Tensor.view](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

---

<link rel="stylesheet" href="{{ site.baseurl }}/assets/css/tensor-layouts.css">
<script src="{{ site.baseurl }}/assets/js/tensor-layouts.js"></script>