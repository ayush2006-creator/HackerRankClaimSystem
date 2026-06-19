import logging
from typing import List, Dict, Any

from schema import ClaimPacket, ClaimDecision, ObjectType

def compute_user_history_flags(user_history: Dict[str, Any]) -> List[str]:
    flags = []
    
    past_claims = user_history.get("past_claim_count")
    rejected = user_history.get("rejected_claim")
    history_flags = user_history.get("history_flags")
    recent = user_history.get("last_90_days_claim_count")
    
    # Check explicitly flagged risk
    if history_flags and str(history_flags).strip().lower() not in ["none", "", "nan"]:
        flags.append("user_history_risk")
        
    # Check claim volume heuristic
    try:
        if past_claims is not None and int(past_claims) > 2:
            # High volume of rejected claims
            if rejected is not None and int(rejected) >= 2:
                if "user_history_risk" not in flags:
                    flags.append("user_history_risk")
            # High recent velocity
            elif recent is not None and int(recent) > 2:
                 if "user_history_risk" not in flags:
                    flags.append("user_history_risk")
    except (ValueError, TypeError):
        pass
        
    return flags

def assemble_decision(
    claim: ClaimPacket,
    stage_a_verdicts: Dict[str, dict],
    stage_b_detections: List[Any],
    stage_c_verdict: Any,
    stage_d_reasoning: dict
) -> ClaimDecision:
    """Merges all pipeline stages into the final Output row schema."""
    
    risk_flags = set()
    
    # 1. Gather Stage A (Quality & Object Match) flags
    valid_image_count = 0
    for img_id, v in stage_a_verdicts.items():
        if v.get("match", False) and v.get("valid_image", True):
            valid_image_count += 1
        
        # Add flags from Stage A
        for f in v.get("risk_flags", []):
            risk_flags.add(f)
            
    # 2. Gather Stage C (Instance Verification) flags
    if stage_c_verdict:
        for f in stage_c_verdict.get("risk_flags", []):
            risk_flags.add(f)
            
    # 3. Gather Stage D (Gemini) flags
    for f in stage_d_reasoning.get("risk_flags", []):
        risk_flags.add(f)
        
    # 4. Gather User History flags
    history_flags = compute_user_history_flags(claim.user_history)
    for f in history_flags:
        risk_flags.add(f)
        
    # Determine Valid Image (were ALL images usable?)
    all_images_valid = True
    for img_id, v in stage_a_verdicts.items():
        if not v.get("valid_image", True):
            all_images_valid = False
            break
            
    # Determine Evidence Standard Met
    instance_failed = stage_c_verdict and stage_c_verdict.get("min_similarity", 1.0) < 0.65 and not stage_c_verdict.get("identity_match")
    
    evidence_standard_met = "true"
    evidence_reason = "Images meet minimum requirements for evaluation."
    
    if valid_image_count == 0:
        evidence_standard_met = "false"
        evidence_reason = "No valid images matching the claimed object were provided."
    elif instance_failed:
        evidence_standard_met = "false"
        evidence_reason = "Images appear to show different object instances; cross-image verification failed."
    elif stage_d_reasoning.get("claim_status") == "not_enough_information":
        evidence_standard_met = "false"
        evidence_reason = stage_d_reasoning.get("justification", "Images lack required context.")
    elif "wrong_object" in risk_flags:
         evidence_standard_met = "false"
         evidence_reason = "Images do not match the claimed object type."

    # Format risk flags for CSV
    final_risk_flags = "none"
    if risk_flags:
        risk_flags.discard("none")
        if risk_flags:
            final_risk_flags = ";".join(sorted(list(risk_flags)))

    return ClaimDecision(
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason=evidence_reason,
        risk_flags=final_risk_flags,
        issue_type=stage_d_reasoning.get("issue_type", "unknown"),
        object_part=stage_d_reasoning.get("object_part", "unknown"),
        claim_status=stage_d_reasoning.get("claim_status", "not_enough_information"),
        claim_status_justification=stage_d_reasoning.get("justification", "Insufficient data to assemble decision."),
        supporting_image_ids=";".join(stage_d_reasoning.get("supporting_image_ids", [])) or "none",
        valid_image="true" if all_images_valid else "false",
        severity=stage_d_reasoning.get("severity", "unknown")
    )
