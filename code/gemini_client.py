"""
gemini_client.py
Shared client routed to Amazon Bedrock running Meta Llama 3.2 Vision.
"""

import os
import io
import re
import logging
from typing import Optional, Any
import boto3

# Bedrock Model configuration
# Defaults to Amazon Nova Pro (v1), which is an active, fully supported vision model.
# Can be overridden via BEDROCK_MODEL_ID in .env
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

logger = logging.getLogger("bedrock_client")

# Thread-safe lazy loaded client
_bedrock_client = None

def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        # Boto3 automatically picks up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
        # and AWS_DEFAULT_REGION from the environment / .env file.
        _bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION
        )
    return _bedrock_client

def generate_content(
    contents: list,
    config: Optional[Any] = None,
    model: str = BEDROCK_MODEL_ID,
) -> str:
    """
    Routes visual-text reasoning content to Amazon Bedrock running Llama 3.2 Vision.
    This accepts a list containing a text prompt and PIL Image objects.
    """
    client = _get_bedrock_client()
    
    text_prompt = ""
    converse_content = []

    # First extract text prompt
    for item in contents:
        if isinstance(item, str):
            text_prompt += item + "\n"
            
    # Add text block to Bedrock Converse content
    if text_prompt:
        converse_content.append({
            "text": text_prompt.strip()
        })

    # Convert PIL Images to Bedrock Converse format
    for item in contents:
        if not isinstance(item, str) and hasattr(item, "save"):  # PIL Image
            try:
                # Resize image slightly to optimize payload size (800x800 for high quality)
                img_copy = item.copy()
                img_copy.thumbnail((1024, 1024))
                
                # Convert to RGB if the image has an alpha channel (RGBA), as JPEG doesn't support transparency
                if img_copy.mode != "RGB":
                    img_copy = img_copy.convert("RGB")
                
                buffered = io.BytesIO()
                img_copy.save(buffered, format="JPEG")
                img_bytes = buffered.getvalue()
                
                converse_content.append({
                    "image": {
                        "format": "jpeg",
                        "source": {
                            "bytes": img_bytes
                        }
                    }
                })
            except Exception as e:
                logger.error(f"Failed to process PIL Image for Bedrock: {e}")

    logger.info(f"Sending request to Amazon Bedrock ({BEDROCK_MODEL_ID}) with {len(converse_content) - 1} images")

    try:
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,  # Force BEDROCK_MODEL_ID directly instead of overridden model parameter
            messages=[
                {
                    "role": "user",
                    "content": converse_content
                }
            ]
        )
        
        message_content = response.get("output", {}).get("message", {}).get("content", [])
        if message_content and "text" in message_content[0]:
            output_text = message_content[0]["text"].strip()
            
            # Clean markdown wrappers if present (e.g. ```json ... ```)
            if output_text.startswith("```"):
                output_text = re.sub(r"^```(?:json)?\n", "", output_text)
                output_text = re.sub(r"\n```$", "", output_text)
            
            return output_text.strip()
        
        return ""
    except Exception as e:
        logger.error(f"Amazon Bedrock Llama 3.2 Vision Converse call failed: {e}")
        # Re-raise standard exception so the pipeline fails safe
        raise
