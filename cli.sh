#!/usr/bin/env bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

alias_config=$(alias lwe 2>/dev/null)
restore_alias() {
  eval "${alias_config}"
}
trap restore_alias EXIT
unalias lwe 2>/dev/null



LWE_CONFIG_DIR="${CURRENT_DIR}/srt_labeler/lwe/config" LWE_DATA_DIR="${CURRENT_DIR}/srt_labeler/lwe/data" lwe "$@"
