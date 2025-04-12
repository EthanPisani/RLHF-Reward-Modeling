# QUEST-RM: Qualitative Understanding and Evaluation Scoring Transformer Reward Model

**Based on ArmoRM**, QUEST-RM is a fine-tuned reward model designed to evaluate roleplay quality in large language models (LLMs). Unlike ArmoRM's original 19 reward objectives, QUEST-RM is streamlined to focus on **six interpretable, roleplay-specific attributes**:
- **Contextual Alignment**
- **Character Consistency**
- **Descriptive Depth**
- **Role-Specific Knowledge**
- **Engagement and Collaboration**
- **Creativity and Emotional Nuance**

QUEST-RM is built on the [FsfairX-LLaMA3-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1) checkpoint and trained through a two-stage pipeline.

---

## Training Pipeline

QUEST-RM follows a two-stage process:
1. **Stage 1: Multi-Objective Reward Learning**
2. **Stage 2: Gating Network and Preference Aggregation**

### Stage 1: Multi-Objective Reward Learning

For each model variant (Model 1–4):

1. **Prepare Embeddings**
```bash
python stage-1_prepare.py \
  --model_path models/FsfairX-LLaMA3-RM-v0.1 \
  --dataset_path QUEST-RM/datasets/modelX \
  --device 0
```

2. **Train Ridge Regressors**
```bash
python stage-1_train.py \
  --model_path models/FsfairX-LLaMA3-RM-v0.1 \
  --dataset_path QUEST-RM/datasets/modelX \
  --output_dir ./stage1_train_modelX/ \
  --embeddings_dir data/ArmoRM/embeddings/FsfairX-LLaMA3-RM-v0.1/modelX
```

### Stage 2: Preference Learning with Gating Network

Trains a gating model to combine multi-objective scores contextually using pairwise preferences and reference datasets. For each model variant (Model 1–4):

```bash
python stage-2_train.py \
  --model_path models/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset QUEST-RM/datasets/modelX \
  --preference_dataset RLHF-Reward-Modeling/armo-rm/pair_data_v2_80K_wsafety/ \
  --reference_dataset ./UltraFeedback-preference-standard \
  --device 0 \
  --eval_reward_bench
```

---

## Key Differences from ArmoRM

| Feature                     | ArmoRM                                | QUEST-RM                                |
|----------------------------|----------------------------------------|------------------------------------------|
| Base Checkpoint            | FsfairX-LLaMA3-RM-v0.1                 | Same                                     |
| # of Attributes            | 19 (Helpfulness, Coherence, etc.)     | 6 (Roleplay-specific traits)             |
| Use Case                   | General LLM alignment tasks            | Roleplay quality evaluation              |
| Evaluation Benchmarks      | RewardBench                            | QUEST-Bench (custom roleplay evaluation) |

---

## Resources

- **Original ArmoRM Paper**: [arXiv:2406.12845](https://arxiv.org/abs/2406.12845)
- **Base Model**: [FsfairX-LLaMA3-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1)
- **Upstream Code**: https://github.com/RLHFlow/RLHF-Reward-Modeling/

---

## Citation

If you build on QUEST-RM, please cite the upstream work on ArmoRM:

```bibtex
@article{ArmoRM,
      title={Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts}, 
      author={Haoxiang Wang and Wei Xiong and Tengyang Xie and Han Zhao and Tong Zhang},
      journal={arXiv preprint arXiv:2406.12845},
}
```

---
