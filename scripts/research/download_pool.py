"""Download public TRAINING sources for the detector research. None of the six benchmark datasets is used."""
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "pool"
FILES = {
    "mosscap": ("Lakera/mosscap_prompt_injection", ["data/train-00000-of-00001-07ae0ed17fa07cc1.parquet"]),
    "inthewild": ("TrustAIRLab/in-the-wild-jailbreak-prompts", ["jailbreak_2023_12_25/train-00000-of-00001.parquet", "regular_2023_12_25/train-00000-of-00001.parquet"]),
    "llmail": ("microsoft/llmail-inject-challenge", ["data/labelled_unique_submissions_phase2.json", "data/emails_for_fp_tests.json"]),
    "wambosec": ("wambosec/prompt-injections", ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]),
    "yanismiraoui": ("yanismiraoui/prompt_injections", ["prompt_injections.csv"]),
    "alespalla": ("alespalla/chatbot_instruction_prompts", ["data/train-00000-of-00001-d3450385c0ae3f98.parquet"]),
    "dolly": ("databricks/databricks-dolly-15k", ["databricks-dolly-15k.jsonl"]),
}
for key, (repo, files) in FILES.items():
    for f in files:
        print(key, hf_hub_download(repo, f, repo_type="dataset", local_dir=ROOT / key))
