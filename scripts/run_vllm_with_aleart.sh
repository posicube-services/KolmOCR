#!/bin/bash

TO="ahnyeonchan@gmail.com"
HOST=$(hostname)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

USE_MSMTP=false
if [[ "$1" == "--use_msmtp" ]]; then
    USE_MSMTP=true
    shift
fi

COMMAND="$@"

send_mail () {
  SUBJECT="[ALERT] vLLM crashed on ${HOST}"
  BODY=$(cat <<EOF
🚨 vLLM 프로세스가 종료되었습니다.

Host      : ${HOST}
StartTime : ${START_TIME}
EndTime   : $(date '+%Y-%m-%d %H:%M:%S')

Command:
${COMMAND}

Check container logs for details.
EOF
)
  if [ "$USE_MSMTP" = true ]; then
    printf "Subject: ${SUBJECT}\n\n${BODY}" | msmtp ${TO}
  else
    echo -e "Subject: ${SUBJECT}\n\n${BODY}" | mail -s "[${HOST}] ALERT vLLM crashed" ${TO}
  fi
}


#printf "Subject: mail test from vessl \n\nThis is test mail\n" | msmtp ahnyeonchan@gmail.com
#echo "This is test mail\n" | mail -s "mail test from 107" ahnyeonchan@gmail.com

# 💥 어떤 이유로든 종료되면 메일 발송
trap send_mail EXIT

# ===== vLLM 실행 (외부 인자 사용) =====
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

eval "$@"

# vessl 사용명령어


# CUDA_VISIBLE_DEVICES=0,1 \
# vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
# --port 8000 \
# --tensor-parallel-size 2 \
# --gpu-memory-utilization 0.9 \
# --max-num-seqs 48
