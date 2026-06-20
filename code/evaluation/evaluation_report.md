# Evaluation Report

## Accuracy Metrics (Sample Set)
- **evidence_standard_met**: 19/20 (95.0%)
- **claim_status**: 14/20 (70.0%)
- **issue_type**: 13/20 (65.0%)
- **object_part**: 17/20 (85.0%)

## Detailed Mismatch Breakdown
| Claim ID | Field | Expected (GT) | Predicted | Justification |
|---|---|---|---|---|
| 1 | severity | medium | high | The image shows a clear dent on the rear bumper of the car, consistent with the customer's claim. |
| 2 | claim_status | supported | contradicted | The user transcript indicates a scratch on the front bumper, but the provided images do not show any visible damage. Specifically, Image img_2 incorrectly localizes a glass shatter issue which is not consistent with the user's claim. |
| 2 | severity | low | none | The user transcript indicates a scratch on the front bumper, but the provided images do not show any visible damage. Specifically, Image img_2 incorrectly localizes a glass shatter issue which is not consistent with the user's claim. |
| 3 | severity | medium | high | The images show a significant crack on the windshield, matching the customer's description of a crack spreading from a small stone impact. The images meet the evidence requirements for a windshield crack. |
| 4 | evidence_standard_met | true | false | The image shows a cracked side mirror, but the user's claim was about the side mirror not sitting properly. The visible damage is a crack, not a misalignment as described. |
| 4 | claim_status | supported | contradicted | The image shows a cracked side mirror, but the user's claim was about the side mirror not sitting properly. The visible damage is a crack, not a misalignment as described. |
| 4 | valid_image | true | false | The image shows a cracked side mirror, but the user's claim was about the side mirror not sitting properly. The visible damage is a crack, not a misalignment as described. |
| 5 | issue_type | scratch | dent | The user described the damage as 'pretty bad', but the image shows only a minor dent on the rear bumper, which contradicts the severity described by the user. |
| 6 | object_part | headlight | unknown | The provided image does not show any specific damage to the car, especially the headlight as mentioned by the customer. The image is a side view of the car with no visible damage. |
| 8 | issue_type | broken_part | scratch | The provided image shows severe damage to the front of the car, including a shattered windshield and extensive body damage, which contradicts the customer's claim of a minor scratch on the hood. The severity and nature of the damage shown are inconsistent with the claim. |
| 8 | object_part | front_bumper | hood | The provided image shows severe damage to the front of the car, including a shattered windshield and extensive body damage, which contradicts the customer's claim of a minor scratch on the hood. The severity and nature of the damage shown are inconsistent with the claim. |
| 8 | valid_image | false | true | The provided image shows severe damage to the front of the car, including a shattered windshield and extensive body damage, which contradicts the customer's claim of a minor scratch on the hood. The severity and nature of the damage shown are inconsistent with the claim. |
| 9 | severity | medium | high | The image clearly shows a cracked laptop screen, aligning with the customer's report of a crack in the display glass. |
| 11 | claim_status | supported | contradicted | The image shows wetness and water droplets on the keyboard, indicating active water exposure rather than a dried stain as described by the customer. |
| 11 | issue_type | stain | water_damage | The image shows wetness and water droplets on the keyboard, indicating active water exposure rather than a dried stain as described by the customer. |
| 13 | severity | medium | high | The image clearly shows a cracked screen with a spider-web pattern, which matches the user's description of marks on the display area. |
| 14 | issue_type | none | dent | The customer claimed physical damage around the trackpad area, but the image shows no visible damage such as cracks, stains, or structural deformations. The trackpad area appears smooth and intact. |
| 15 | claim_status | supported | contradicted | The image shows the package with a visible corner, but there is no clear indication of crushing or damage as described by the customer. The corner appears intact. |
| 15 | severity | medium | none | The image shows the package with a visible corner, but there is no clear indication of crushing or damage as described by the customer. The corner appears intact. |
| 16 | claim_status | supported | contradicted | The user claimed that the seal was torn, suggesting the package was opened. However, the images provided show the package in a sealed condition with no visible damage to the seal. Therefore, the claim is contradicted by the visual evidence. |
| 16 | severity | medium | none | The user claimed that the seal was torn, suggesting the package was opened. However, the images provided show the package in a sealed condition with no visible damage to the seal. Therefore, the claim is contradicted by the visual evidence. |
| 17 | issue_type | water_damage | stain | The image shows a visible discoloration and residue mark on the package side, indicating a stain rather than active water damage. The customer's description of a 'wet-looking stain' aligns with the visual evidence of a dried liquid mark on the package surface. |
| 18 | claim_status | not_enough_information | contradicted | The images do not provide enough information to confirm the customer's claim about missing contents. Image 1 shows a box filled with crumpled paper, which could be packing material, and Image 2 shows an unopened box. There is no clear evidence of missing contents. |
| 19 | issue_type | unknown | crushed_packaging | The image provided shows a close-up of a can, not the shipping box. There is no clear evidence of the claimed damage to the shipping box. |
| 19 | object_part | unknown | box | The image provided shows a close-up of a can, not the shipping box. There is no clear evidence of the claimed damage to the shipping box. |
| 19 | severity | low | unknown | The image provided shows a close-up of a can, not the shipping box. There is no clear evidence of the claimed damage to the shipping box. |
| 20 | issue_type | none | torn_packaging | The images do not show any visible damage to the package seal or any other part of the package. Both images appear to show the package in good condition without any tears or damage. |

## Operational Analysis
- **Stage A (Object Verification)**: Local CLIP inference. Cost: Free. Call count: 1 per image.
- **Stage B (Damage Detection)**: Roboflow API / Local fallback. Call count: 1 per valid image.
- **Stage C (Instance Verification)**: Local DINOv2 inference. Cost: Free. Call count: 1 per valid image if >1 image.
- **Stage D (Reasoning)**: Amazon Bedrock (Nova Pro). Call count: Exactly 1 per claim. All images and signals sent in one batch.

This architecture significantly reduces API calls compared to the naive approach by moving object verification and instance matching to local, free models, and restricting the LLM to a single reasoning step per claim.
