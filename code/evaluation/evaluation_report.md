# Evaluation Report

## Accuracy Metrics (Sample Set)
- **evidence_standard_met**: 16/20 (80.0%)
- **claim_status**: 10/20 (50.0%)
- **issue_type**: 12/20 (60.0%)
- **object_part**: 17/20 (85.0%)

## Operational Analysis
- **Stage A (Object Verification)**: Local CLIP inference. Cost: Free. Call count: 1 per image.
- **Stage B (Damage Detection)**: Roboflow API / Local fallback. Call count: 1 per valid image.
- **Stage C (Instance Verification)**: Local DINOv2 inference. Cost: Free. Call count: 1 per valid image if >1 image.
- **Stage D (Reasoning)**: Gemini Flash Lite. Call count: Exactly 1 per claim. All images and signals sent in one batch.

This architecture significantly reduces API calls compared to the naive approach by moving object verification and instance matching to local, free models, and restricting the LLM to a single reasoning step per claim.
