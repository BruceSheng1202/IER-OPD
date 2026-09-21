# Training patches

Apply these patches to the matching source checkouts before installation.

| Patch | Source revision |
|---|---|
| `sglang.patch` | SGLang `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2` |
| `megatron.patch` | Megatron-LM `3714d81d418c9f1bca4594fc35f9e8289f652862` |

Replace the paths below with your local checkout and repository paths:

```bash
git -C /path/to/sglang apply /path/to/IER-OPD/third_party/patches/sglang.patch
git -C /path/to/Megatron-LM apply /path/to/IER-OPD/third_party/patches/megatron.patch
```

Apply each patch once. Dependency repositories and revisions are listed in
[UPSTREAM.json](../UPSTREAM.json).
