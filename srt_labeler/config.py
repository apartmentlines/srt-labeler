import os
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LWE_CONFIG_DIR = BASE_DIR / "lwe" / "config"
LWE_DATA_DIR = BASE_DIR / "lwe" / "data"


def load_configuration(args: argparse.Namespace) -> tuple[str, str, str]:
    api_key = args.api_key or os.environ.get("SRT_LABELER_API_KEY", "")
    file_api_key = args.file_api_key or os.environ.get("SRT_LABELER_FILE_API_KEY", "")
    domain = args.domain or os.environ.get("SRT_LABELER_DOMAIN", "")
    if not api_key or not domain:
        raise ValueError(
            "API key and domain must be provided either via CLI arguments or environment variables."
        )
    return api_key, file_api_key, domain


def set_environment_variables(api_key: str | None, file_api_key: str | None, domain: str | None) -> None:
    if api_key:
        os.environ["SRT_LABELER_API_KEY"] = api_key
    if file_api_key:
        os.environ["SRT_LABELER_FILE_API_KEY"] = file_api_key
    if domain:
        os.environ["SRT_LABELER_DOMAIN"] = domain
    os.environ["LWE_CONFIG_DIR"] = str(LWE_CONFIG_DIR)
    os.environ["LWE_DATA_DIR"] = str(LWE_DATA_DIR)
