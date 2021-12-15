# Clip on Image and Hashes

*NOTE: This project investigated an idea that turned out not to work. However, the code here may be useful for others.*

The hypothesis is that we can train a multimodal embedding using CLIP between an image and its SHA256 hash. If this is achievable, then for downstream tasks, we only need to store the hash of images, thus protecting the semantic content. Hash functions are known to be random but potentially there exists a manifold (guided by the image modality) where "semantically similar hashes" are grouped together. This would be interesting for cryptographic purposes. 

However, in our experiments, we found that the learned hash embeddings were not useful: they predicted labels at chance accuracy. We tried different image embeddings (ResNet18/50 pretrained or from scratch) as well as different text embeddings (DeBERTa, RoBERTa), but found the same result. 

### Dependencies

Beyond the usual dependencies (PyTorch, Huggingface, Lightning, etc.), we also used Microsoft's [DeBERTa](https://github.com/microsoft/DeBERTa) which needs to be installed separately.

### Resources

We based several implementation on code from the following links.

- https://pythonrepo.com/repo/moein-shariatnia-OpenAI-CLIP-python-deep-learning
- https://github.com/mlfoundations/open_clip
