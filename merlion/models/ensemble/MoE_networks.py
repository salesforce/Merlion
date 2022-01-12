#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[deep-learning]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)


class DimDropout(torch.nn.Module):
    """
    given input of dim: BS x Seq x dims, randomly pick and mask some of the dims of each sample s.t. all time steps
    of each sample have the same dims mask and each sample has a different dims mask
    """

    def __init__(self, p, replicate_dim=1):
        """ """
        super().__init__()
        self.p = p
        self.replicate_dim = replicate_dim

    def forward(self, x):
        """ """
        mask = 1
        if self.p > 0:
            if self.replicate_dim == 1:
                mask = (1.0 - self.p) * torch.ones(x.size(0), 1, x.size(2))
                # create masks of dim batch_size x 1 x feat_dim
            elif self.replicate_dim == 2:
                mask = (1.0 - self.p) * torch.ones(x.size(0), x.size(1), 1)
                # create masks of dim batch_size x lookback_len x 1
            mask = torch.bernoulli(mask).cuda()
        return x * mask


class MLP(nn.Module):
    def __init__(self, inp_dim, hid_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        hd = qkv[0].size(-1)

        def myrearrange(h):
            def fn(t):
                t = t.reshape(t.size(0), t.size(1), h, -1)
                t = t.permute(0, 2, 1, 3)
                return t

            return fn

        q, k, v = map(myrearrange(h=h), qkv)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        def myrearrange(t):
            t = t.permute(0, 2, 1, 3)
            t = t.reshape(t.size(0), t.size(1), -1)
            return t

        out = myrearrange(out)

        out = self.to_out(out)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        lookback_len,
        nexperts,
        output_dim,
        nfree_experts=0,
        hid_dim=128,
        depth=2,
        heads=8,
        mlp_dim=32,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        dim_dropout=0,
        time_step_dropout=0.0,
    ):
        super().__init__()

        self.nfree_experts = nfree_experts
        self.output_dim = output_dim
        self.dim_dropout = DimDropout(dim_dropout)
        self.time_step_dropout = DimDropout(time_step_dropout, replicate_dim=2)
        self.inp_Dropout = nn.Dropout(dim_dropout)
        self.inp_embedding = MLP(input_dim, hid_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, lookback_len + 1, hid_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerLayer(hid_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hid_dim), nn.Linear(hid_dim, (nexperts + nfree_experts) * output_dim)
        )
        if nfree_experts > 0:
            self.expert_val = nn.Parameter(torch.randn(nfree_experts, output_dim))

    def forward(self, x):

        if self.training:
            x = self.dim_dropout(x)
            x = self.time_step_dropout(x)
        x = self.inp_embedding(x)

        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        # ensures that pos_embedding works even when input seq is shorter than lookback_len
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)

        logits = self.mlp_head(x)
        logits = logits.reshape(logits.size(0), -1, self.output_dim)  # B x total num experts x forecast len
        return (logits, self.expert_val) if self.nfree_experts > 0 else (logits, None)

    def drop_inp(self, x, p):
        if p > 0:
            dim_dropout = DimDropout(p)
            x = dim_dropout(x)
        return x


class LRScheduler:
    def __init__(self, lr_i, lr_f, nsteps, optimizer):
        self.step_ = 0
        self.nsteps = nsteps
        self.lr_i, self.lr_f = lr_i, lr_f
        self.optimizer = optimizer

    def step(self):
        self.step_ += 1
        if self.step_ <= self.nsteps:
            self.lr = self.lr_i + (self.lr_f - self.lr_i) * float(self.step_) / self.nsteps
        else:
            self.lr = self.lr_f
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr
