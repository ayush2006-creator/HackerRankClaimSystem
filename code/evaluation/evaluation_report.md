# Evaluation Report

## Accuracy Metrics (Sample Set)
- **evidence_standard_met**: 16/20 (80.0%)
- **claim_status**: 15/20 (75.0%)
- **issue_type**: 11/20 (55.0%)
- **object_part**: 15/20 (75.0%)

## Detailed Mismatch Breakdown
| Claim ID | Field | Expected (GT) | Predicted | Justification |
|---|---|---|---|---|
| 1 | severity | medium | high | The image shows a significant dent in the rear bumper of the car, consistent with the customer's description of the damage. |
| 2 | evidence_standard_met | true | false | The user transcript indicates a scratch on the front bumper, but the provided images do not clearly show the damage due to the angle and clarity. Image img_1 is marked as blurry. |
| 2 | claim_status | supported | not_enough_information | The user transcript indicates a scratch on the front bumper, but the provided images do not clearly show the damage due to the angle and clarity. Image img_1 is marked as blurry. |
| 2 | severity | low | unknown | The user transcript indicates a scratch on the front bumper, but the provided images do not clearly show the damage due to the angle and clarity. Image img_1 is marked as blurry. |
| 3 | issue_type | crack | glass_shatter | The provided images clearly show a shattered windshield, matching the customer's claim. Image img_1 shows the shattered glass with high confidence, fulfilling the evidence requirement for a cracked or broken glass component. |
| 3 | severity | medium | high | The provided images clearly show a shattered windshield, matching the customer's claim. Image img_1 shows the shattered glass with high confidence, fulfilling the evidence requirement for a cracked or broken glass component. |
| 4 | evidence_standard_met | true | false | The image is blurry and does not clearly show the damage to the side mirror. More detailed images are required to assess the claim properly. |
| 4 | claim_status | supported | not_enough_information | The image is blurry and does not clearly show the damage to the side mirror. More detailed images are required to assess the claim properly. |
| 4 | valid_image | true | false | The image is blurry and does not clearly show the damage to the side mirror. More detailed images are required to assess the claim properly. |
| 4 | severity | medium | unknown | The image is blurry and does not clearly show the damage to the side mirror. More detailed images are required to assess the claim properly. |
| 5 | issue_type | scratch | dent | The transcript indicates the customer claims significant damage to the rear bumper, but the images show only minor surface marks and a dent, not severe damage as described. |
| 6 | object_part | headlight | unknown | The provided image does not show the headlight or any specific damaged part as claimed by the customer. The image shows the side of the car with no visible damage. |
| 8 | issue_type | broken_part | scratch | The provided image shows a severely damaged car with extensive structural damage to the front bumper and body, but no scratch is visible on the hood as claimed by the user. |
| 8 | object_part | front_bumper | hood | The provided image shows a severely damaged car with extensive structural damage to the front bumper and body, but no scratch is visible on the hood as claimed by the user. |
| 8 | valid_image | false | true | The provided image shows a severely damaged car with extensive structural damage to the front bumper and body, but no scratch is visible on the hood as claimed by the user. |
| 9 | issue_type | crack | glass_shatter | The image shows a shattered laptop screen, which matches the customer's report of a cracked display glass. The screen is clearly visible and the damage is consistent with a glass shatter. |
| 9 | severity | medium | high | The image shows a shattered laptop screen, which matches the customer's report of a cracked display glass. The screen is clearly visible and the damage is consistent with a glass shatter. |
| 10 | evidence_standard_met | true | false | The customer claims the hinge area is broken and the screen wobbles. However, the images provided do not clearly show the hinge area or any visible damage to it. Image img_1 is blurry and does not provide sufficient detail, while image img_2 does not show the hinge area at all. |
| 10 | claim_status | supported | not_enough_information | The customer claims the hinge area is broken and the screen wobbles. However, the images provided do not clearly show the hinge area or any visible damage to it. Image img_1 is blurry and does not provide sufficient detail, while image img_2 does not show the hinge area at all. |
| 10 | severity | medium | unknown | The customer claims the hinge area is broken and the screen wobbles. However, the images provided do not clearly show the hinge area or any visible damage to it. Image img_1 is blurry and does not provide sufficient detail, while image img_2 does not show the hinge area at all. |
| 11 | issue_type | stain | water_damage | The image shows water droplets on the keyboard, but there is no clear indication of a stain or sticky keys as described by the customer. The automated damage localization did not detect any damage. |
| 11 | severity | medium | low | The image shows water droplets on the keyboard, but there is no clear indication of a stain or sticky keys as described by the customer. The automated damage localization did not detect any damage. |
| 13 | issue_type | crack | glass_shatter | The image clearly shows a shattered laptop screen, aligning with the customer's claim of screen damage. |
| 13 | severity | medium | high | The image clearly shows a shattered laptop screen, aligning with the customer's claim of screen damage. |
| 16 | object_part | seal | box | The images show clear evidence of torn packaging on the box, aligning with the customer's claim of the seal being torn. |
| 17 | issue_type | water_damage | stain | The image shows clear evidence of a stain on the exterior of the package, aligning with the customer's claim of water damage. |
| 17 | object_part | package_side | box | The image shows clear evidence of a stain on the exterior of the package, aligning with the customer's claim of water damage. |
| 18 | claim_status | not_enough_information | contradicted | The images do not provide clear evidence of the missing contents as claimed by the user. The images show an opened package but do not confirm the absence of the specific item claimed to be missing. |
| 19 | evidence_standard_met | true | false | The image provided does not clearly show the exterior of the package in a manner that allows for a thorough inspection of the claimed damage. The image is too blurry and does not meet the requirement for clear visibility of the package exterior. |
| 19 | claim_status | contradicted | not_enough_information | The image provided does not clearly show the exterior of the package in a manner that allows for a thorough inspection of the claimed damage. The image is too blurry and does not meet the requirement for clear visibility of the package exterior. |
| 19 | issue_type | unknown | crushed_packaging | The image provided does not clearly show the exterior of the package in a manner that allows for a thorough inspection of the claimed damage. The image is too blurry and does not meet the requirement for clear visibility of the package exterior. |
| 19 | object_part | unknown | box | The image provided does not clearly show the exterior of the package in a manner that allows for a thorough inspection of the claimed damage. The image is too blurry and does not meet the requirement for clear visibility of the package exterior. |
| 19 | severity | low | unknown | The image provided does not clearly show the exterior of the package in a manner that allows for a thorough inspection of the claimed damage. The image is too blurry and does not meet the requirement for clear visibility of the package exterior. |
| 20 | issue_type | none | torn_packaging | The user claimed that the package arrived with a torn seal. However, the images show no visible damages to the seal or any other part of the package. Therefore, the claim is contradicted based on the visual evidence provided. |

## Operational Analysis
- **Stage A (Object Verification)**: Local CLIP inference. Cost: Free. Call count: 1 per image.
- **Stage B (Damage Detection)**: Roboflow API / Local fallback. Call count: 1 per valid image.
- **Stage C (Instance Verification)**: Local DINOv2 inference. Cost: Free. Call count: 1 per valid image if >1 image.
- **Stage D (Reasoning)**: Amazon Bedrock (Nova Pro). Call count: Exactly 1 per claim. All images and signals sent in one batch.

This architecture significantly reduces API calls compared to the naive approach by moving object verification and instance matching to local, free models, and restricting the LLM to a single reasoning step per claim.
