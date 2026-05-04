"""ACE (Adversarial Correspondence Embedding) Stage 2 components.

Reuses frozen 70 Stage A VQ-VAEs (model/moreflow/) for motion prior + decoder.
Adds Generator + Discriminator + paper-faithful adversarial training.

References:
  papers/2305.14792.pdf — Li et al., SIGGRAPH Asia 2023
  refine-logs/ACE_DESIGN_V3.md
"""
