import os
from inference_sdk import InferenceHTTPClient

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

try:
    result = client.run_workflow(
        workspace_name="ayushs-workspace-zfnzn",
        workflow_id="laptop-damage", # trying a guess
        images={"image": "dataset/images/sample/claim_1.jpg"},
        parameters={"classes": "body-damage, display-damage, keyboard-damage"}
    )
    print("Success with laptop-damage:", result)
except Exception as e:
    print("Failed with laptop-damage:", e)

try:
    result = client.run_workflow(
        workspace_name="ayushs-workspace-zfnzn",
        workflow_id="zero-shot-object-detection", # trying standard
        images={"image": "dataset/images/sample/claim_1.jpg"},
        parameters={"classes": "body-damage, display-damage, keyboard-damage"}
    )
    print("Success with zero-shot-object-detection:", result)
except Exception as e:
    print("Failed with zero-shot-object-detection:", e)
