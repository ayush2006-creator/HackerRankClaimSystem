import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Model Names / IDs
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DINO_MODEL_NAME = "facebook/dinov2-small"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Roboflow Universe Model IDs (workspace/project/version)
# These may need to be tuned based on what's available on Universe
ROBOFLOW_CAR_MODEL = "general-segmentation-api-3"
ROBOFLOW_LAPTOP_MODEL = "laptop-damage-detection-testing/6"
ROBOFLOW_PACKAGE_MODEL = "damage-package-detection/5"

# Thresholds
OBJECT_CONFIDENCE_FLOOR = 0.5
DAMAGE_CONFIDENCE_FLOOR = 0.4
DINO_SIMILARITY_THRESHOLD = 0.65
BLUR_LAPLACIAN_THRESHOLD = 15.0

# Quality Check Thresholds
LOW_LIGHT_THRESHOLD = 40.0

# Paths
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
SAMPLE_CLAIMS_CSV = os.path.join(DATASET_DIR, "sample_claims.csv")
TEST_CLAIMS_CSV = os.path.join(DATASET_DIR, "claims.csv")
USER_HISTORY_CSV = os.path.join(DATASET_DIR, "user_history.csv")
EVIDENCE_REQ_CSV = os.path.join(DATASET_DIR, "evidence_requirements.csv")
