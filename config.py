"""Central configuration for the prerequisite question bank pipeline.

All settings are hardcoded here. Stage modules import this module for
LLM endpoint details, concurrency/retry policy, similarity thresholds,
and the canonical intermediate/output file paths.
"""
from pathlib import Path

# --- LLM endpoint (OpenAI-compatible vLLM server) -------------------------
LLM_BASE_URL = "http://localhost:30000/v1"
LLM_API_KEY = "vllm"
LLM_MODEL = (
    "/fsx2/opensource-models/hub/models--openai--gpt-oss-120b/"
    "snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/"
)

# --- Concurrency / retry / timeouts ---------------------------------------
MAX_CONCURRENT = 8
RETRY_ATTEMPTS = 3
TIMEOUT_SECONDS = 60

# --- Similarity thresholds (token-level Jaccard, whitespace tokenization) --
SIMILARITY_DEDUP_THRESHOLD = 0.92
SIMILARITY_CONTAMINATION_THRESHOLD = 0.85

# --- Project layout ---------------------------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent
LOG_FILE = PIPELINE_DIR / "pipeline.log"
README_PATH = PIPELINE_DIR / "README.md"

# --- Domains -----------------------------------------------------------------
DOMAINS = ("coding", "reasoning")

# --- Per-domain file paths -----------------------------------------------------
def reasoning_file(domain: str) -> Path:
    """Stage 1 output: source records + grounded reasoning trace."""
    return PIPELINE_DIR / f"{domain}_with_reasoning.json"


def raw_concepts_file(domain: str) -> Path:
    """Stage 2/3 output: source records + raw (and normalized) concepts."""
    return PIPELINE_DIR / f"{domain}_raw_concepts.json"


def taxonomy_file(domain: str) -> Path:
    """Stage 3 output: canonical taxonomy + raw->canonical merge map."""
    return PIPELINE_DIR / f"{domain}_taxonomy.json"


def failure_modes_file(domain: str) -> Path:
    """Stage 4 output: source records + merged two-pass failure modes."""
    return PIPELINE_DIR / f"{domain}_with_failure_modes.json"


def concept_graph_file(domain: str) -> Path:
    """Stage 5 output: prerequisite DAG + transitive prerequisites (depth 3)."""
    return PIPELINE_DIR / f"{domain}_concept_graph.json"


def questions_raw_file(domain: str) -> Path:
    """Stage 6 output: generated diagnostic questions (pre-validation)."""
    return PIPELINE_DIR / f"{domain}_questions_raw.json"


def questions_validated_file(domain: str) -> Path:
    """Stage 7 output: all questions with validation_passed/reason."""
    return PIPELINE_DIR / f"{domain}_questions_validated.json"


def final_output_file(domain: str) -> Path:
    """Stage 8 output: validated-only final question bank."""
    return PIPELINE_DIR / f"{domain}_prerequisite_questions.json"
