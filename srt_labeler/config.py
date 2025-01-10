import os
import argparse


def load_configuration(args: argparse.Namespace) -> tuple[str, str]:
    api_key = args.api_key or os.environ.get("SRT_LABELER_API_KEY")
    domain = args.domain or os.environ.get("SRT_LABELER_DOMAIN")
    if not api_key or not domain:
        raise ValueError(
            "API key and domain must be provided either via CLI arguments or environment variables."
        )
    return api_key, domain


def set_environment_variables(api_key: str, domain: str) -> None:
    os.environ["SRT_LABELER_API_KEY"] = api_key
    os.environ["SRT_LABELER_DOMAIN"] = domain
