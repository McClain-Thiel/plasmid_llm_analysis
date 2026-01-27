#!/bin/bash -euo pipefail
# Extract all sequence lines from FASTA (remove header, concatenate all lines), uppercase
PROMPT_SEQ=$(grep -v "^>" GFP_cassette.fasta | tr -d '\n' | tr 'a-z' 'A-Z')

# capture process environment
set +u
set +e
cd "$NXF_TASK_WORKDIR"

nxf_eval_cmd() {
    {
        IFS=$'\n' read -r -d '' "${1}";
        IFS=$'\n' read -r -d '' "${2}";
        (IFS=$'\n' read -r -d '' _ERRNO_; return ${_ERRNO_});
    } < <((printf '\0%s\0%d\0' "$(((({ shift 2; "${@}"; echo "${?}" 1>&3-; } | tr -d '\0' 1>&4-) 4>&2- 2>&1- | tr -d '\0' 1>&4-) 3>&1- | exit "$(cat)") 4>&1-)" "${?}" 1>&2) 2>&1)
}

echo '' > .command.env
#
echo PROMPT_SEQ="${PROMPT_SEQ[@]}" >> .command.env
echo /PROMPT_SEQ/ >> .command.env
