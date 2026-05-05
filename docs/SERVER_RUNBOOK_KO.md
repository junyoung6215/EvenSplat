# NoPoSplat OLP-H 서버 실행 Runbook

이 문서는 서버에 다시 접속해서 NoPoSplat OLP-H 실험을 이어서 확인하거나
재실행하기 위한 명령어 모음입니다.

주의:

- 서버 IP, ID, PW, 개인 SSH key는 이 repo에 저장하지 않습니다.
- 실제 접속 정보는 개인 메모에만 보관합니다.
- 공유 서버에서는 workspace, conda env, output을 실험 slot 단위로 분리합니다.

## 0. 기억해야 하는 이름 치트시트

서버 접속 정보는 개인 메모에만 저장합니다.

```text
SSH alias:
<server-alias>

SSH 직접 접속 형식:
ssh -p <SSH_PORT> <USER>@<SERVER_HOST>
```

이번 실험에서 고정해서 쓰는 이름:

```text
tmux session:
olp_h_re10k_slot0

workspace:
/workspace/olp_h_re10k_slot0

repo:
/workspace/olp_h_re10k_slot0/NoPoSplat

conda env:
noposplat-olp-cuda118-slot0

output root:
outputs/olp_re10k

main log:
outputs/olp_re10k/pilot_nohup.log

env check:
outputs/olp_re10k/env_check/
```

중요 run / output 이름:

```text
official baseline eval:
re10k_official_eval_stats

OLP zero-shot diagnostic:
re10k_olp_zeroshot

OLP fine-tuning:
re10k_olp_ft

OLP fine-tuned eval:
re10k_olp_ft_eval

overnight controller:
outputs/olp_re10k/overnight/<SESSION_ID>/
```

중요 checkpoint / stats 경로:

```text
official checkpoint:
pretrained_weights/re10k.ckpt

OLP fine-tuned checkpoint:
outputs/olp_re10k/re10k_olp_ft/<RUN_ID>/checkpoints/last.ckpt

baseline stats:
outputs/olp_re10k/re10k_official_eval_stats/benchmark.json
outputs/olp_re10k/re10k_official_eval_stats/olp_stats_summary.json
outputs/olp_re10k/re10k_official_eval_stats/olp_stats_per_scene.csv

OLP stats:
outputs/olp_re10k/re10k_olp_ft_eval/benchmark.json
outputs/olp_re10k/re10k_olp_ft_eval/olp_stats_summary.json
outputs/olp_re10k/re10k_olp_ft_eval/olp_stats_per_scene.csv
```

자주 쓰는 script 이름:

```text
환경 생성:
scripts/create_cuda118_env.sh

환경 검증:
scripts/check_cuda118_env.py

checkpoint 다운로드:
scripts/download_re10k_checkpoint.sh

RE10K 데이터셋 준비:
scripts/setup_re10k_dataset.sh

smoke run:
scripts/run_re10k_smoke.sh

오늘 pilot 전체 실행:
scripts/run_re10k_pilot_overnight.sh

stage별 overnight 실행:
scripts/run_re10k_overnight_protocol.sh

최종 통계 수집:
scripts/run_re10k_collect_stats.sh
```

중요 환경변수:

```text
CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=8
OOM이면 BATCH_SIZE=4
LR=2e-4
MAX_STEPS=18751
OLP_SEED=0
BASELINE_CKPT=./pretrained_weights/re10k.ckpt
```

CUDA extension을 다시 빌드할 때 기억할 이름:

```text
CUDA_HOME=$CONDA_PREFIX
CUDA_PATH=$CONDA_PREFIX
CC=/usr/bin/gcc-11
CXX=/usr/bin/g++-11
TORCH_CUDA_ARCH_LIST=8.0
MAX_JOBS=4
```

## 1. 로컬 Mac에서 서버 접속

SSH alias를 설정해둔 경우:

```bash
ssh <server-alias>
```

alias 없이 직접 접속하는 경우:

```bash
ssh -p <SSH_PORT> <USER>@<SERVER_HOST>
```

처음 접속 시 host key 확인 질문이 나오면, 서버 정보가 맞는지 확인한 뒤
정확히 `yes`를 입력합니다.

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

여기서 `no`를 입력하면 접속이 중단됩니다.

## 2. 서버 기본 상태 확인

서버에 접속하면 먼저 아래를 확인합니다.

```bash
pwd
hostname
whoami
nvidia-smi
df -h
```

확인할 것:

- GPU가 보이는지.
- 다른 프로세스가 GPU를 사용 중인지.
- `/home/data` 또는 workspace 쪽 여유 용량이 충분한지.
- Docker/container 내부에서는 할당된 GPU만 보일 수 있습니다.

## 3. tmux 사용

tmux가 없으면 root 권한에서 설치합니다.

```bash
apt update
apt install -y tmux
tmux -V
```

새 세션 시작:

```bash
tmux new -s olp_h_re10k_slot0
```

세션에서 빠져나오기:

```text
Ctrl + b
d
```

다시 접속:

```bash
tmux ls
tmux attach -t olp_h_re10k_slot0
```

## 4. 작업 위치

공유 서버에서는 사람 이름 대신 실험 slot 이름을 사용합니다.

```bash
mkdir -p /workspace/olp_h_re10k_slot0
cd /workspace/olp_h_re10k_slot0
```

repo가 없는 경우:

```bash
git clone https://github.com/junyoung6215/capstone_project.git NoPoSplat
cd NoPoSplat
```

이미 repo가 있는 경우:

```bash
cd /workspace/olp_h_re10k_slot0/NoPoSplat
```

코드 확인:

```bash
git log -1 --oneline
ls scripts/create_cuda118_env.sh
ls scripts/check_cuda118_env.py
ls scripts/run_re10k_pilot_overnight.sh
```

GitHub clone이 어렵고 로컬 Mac에서 복사해야 하는 경우, 아래 명령은 서버가 아니라
Mac 로컬 터미널에서 실행합니다.

```bash
rsync -avz --progress \
  --exclude ".git/" \
  --exclude "outputs/" \
  --exclude "datasets/" \
  --exclude "pretrained_weights/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  -e "ssh -p <SSH_PORT>" \
  "/Users/kwon_junyoung/Desktop/4rd_1st/캡스톤디자인/NoPoSplat/" \
  <USER>@<SERVER_HOST>:/workspace/olp_h_re10k_slot0/NoPoSplat/
```

## 5. Conda 준비

서버에서 conda가 로드되어 있는지 확인합니다.

```bash
which conda
```

Miniconda를 직접 설치한 경우, 매번 새 shell에서 아래를 먼저 실행합니다.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
```

conda가 없으면 설치합니다.

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
source /root/miniconda3/etc/profile.d/conda.sh
conda --version
```

## 6. CUDA 11.8 환경 생성

처음 한 번만 실행합니다.

```bash
cd /workspace/olp_h_re10k_slot0/NoPoSplat
source /root/miniconda3/etc/profile.d/conda.sh

ENV_NAME=noposplat-olp-cuda118-slot0 \
bash scripts/create_cuda118_env.sh
```

공유 서버에서 slot을 바꾸는 경우 env 이름도 같이 바꿉니다.

```bash
ENV_NAME=noposplat-olp-cuda118-slot1 \
bash scripts/create_cuda118_env.sh
```

## 7. 기존 환경 다시 활성화

나중에 다시 접속했을 때 가장 자주 쓰는 명령입니다.

```bash
cd /workspace/olp_h_re10k_slot0/NoPoSplat
source /root/miniconda3/etc/profile.d/conda.sh
conda activate noposplat-olp-cuda118-slot0
```

CUDA extension을 다시 빌드해야 할 때는 아래 환경변수도 고정합니다.

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
```

## 8. 환경 검증

```bash
python scripts/check_cuda118_env.py --output-dir outputs/olp_re10k/env_check
```

수동 확인:

```bash
which nvcc
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization; print('diff_gaussian_rasterization OK')"
```

기대:

```text
torch == 2.1.2+cu118
torch.version.cuda == 11.8
torch.cuda.is_available() == True
nvcc release 11.8
diff_gaussian_rasterization OK
```

## 9. Checkpoint 준비

공식 RE10K checkpoint 다운로드:

```bash
bash scripts/download_re10k_checkpoint.sh
ls -lh pretrained_weights/re10k.ckpt
```

## 10. RE10K 데이터셋 연결

기본 코드가 보는 위치:

```text
datasets/re10k
```

서버에서 데이터셋 위치 찾기:

```bash
find /home/data -maxdepth 5 -type d -iname "*re10k*" 2>/dev/null
find /workspace -maxdepth 5 -type d -iname "*re10k*" 2>/dev/null
```

주의:

- `pretrained_weights/re10k.ckpt`는 모델 가중치입니다.
- 데이터셋은 별도로 필요합니다.
- NoPoSplat은 raw RealEstate10K 영상이 아니라 processed torch chunk 형식을
  기대합니다.

필요한 구조:

```text
datasets/re10k/
  train/
    index.json
    *.torch
  test/
    index.json
    *.torch
```

실제 경로를 찾은 뒤 symlink 생성:

```bash
mkdir -p datasets
ln -s /실제/RE10K/경로 datasets/re10k
ls datasets/re10k
```

또는 setup script를 사용합니다.

기존 processed dataset 경로를 연결:

```bash
RE10K_SOURCE_DIR=/실제/RE10K/경로 \
bash scripts/setup_re10k_dataset.sh
```

직접 다운로드 가능한 archive URL이 있는 경우:

```bash
RE10K_DATA_URL=https://.../processed_re10k.zip \
bash scripts/setup_re10k_dataset.sh
```

이미 `datasets/re10k`가 있다고 생각되면 구조만 검증:

```bash
bash scripts/setup_re10k_dataset.sh
```

이미 잘못된 symlink가 있으면 지운 뒤 다시 만듭니다.

```bash
rm datasets/re10k
ln -s /실제/RE10K/경로 datasets/re10k
```

## 11. Smoke Run

full run 전에 반드시 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=1 MAX_STEPS=2 CKPT_EVERY_N_STEPS=1 \
bash scripts/run_re10k_smoke.sh
```

smoke run이 실패하면 pilot run으로 넘어가지 않습니다.

## 12. Pilot Run

오늘 GPU를 단독 사용한다면 batch 8로 시작합니다.

```bash
mkdir -p outputs/olp_re10k

CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=8 \
nohup bash scripts/run_re10k_pilot_overnight.sh \
  > outputs/olp_re10k/pilot_nohup.log 2>&1 &
```

OOM이 나면 batch 4로 낮춥니다.

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=4 \
nohup bash scripts/run_re10k_pilot_overnight.sh \
  > outputs/olp_re10k/pilot_nohup.log 2>&1 &
```

## 13. 실행 상태 확인

로그 확인:

```bash
tail -f outputs/olp_re10k/pilot_nohup.log
```

GPU 확인:

```bash
nvidia-smi
```

python process 확인:

```bash
ps -eo pid,user,etime,cmd | grep python | grep -v grep
```

overnight manifest 확인:

```bash
ls outputs/olp_re10k/overnight/
cat outputs/olp_re10k/overnight/<SESSION_ID>/manifest.txt
```

stage별 log 확인:

```bash
ls outputs/olp_re10k/overnight/<SESSION_ID>/logs/
tail -f outputs/olp_re10k/overnight/<SESSION_ID>/logs/*.log
```

## 14. 실험 중단

먼저 PID를 확인합니다.

```bash
ps -eo pid,user,etime,cmd | grep python | grep -v grep
```

내 실험 PID만 골라 종료합니다.

```bash
kill <PID>
```

강제 종료는 마지막 수단입니다.

```bash
kill -9 <PID>
```

## 15. 나중에 다시 접속해서 확인하는 최소 명령

```bash
ssh <server-alias>
tmux attach -t olp_h_re10k_slot0

cd /workspace/olp_h_re10k_slot0/NoPoSplat
source /root/miniconda3/etc/profile.d/conda.sh
conda activate noposplat-olp-cuda118-slot0

tail -f outputs/olp_re10k/pilot_nohup.log
nvidia-smi
```

tmux 세션이 없으면:

```bash
tmux new -s olp_h_re10k_slot0
```

## 16. 백업할 산출물

최소 백업:

```text
outputs/olp_re10k/pilot_nohup.log
outputs/olp_re10k/env_check/
outputs/olp_re10k/overnight/<SESSION_ID>/manifest.txt
outputs/olp_re10k/overnight/<SESSION_ID>/logs/

outputs/olp_re10k/re10k_olp_ft/<RUN_ID>/checkpoints/last.ckpt

outputs/olp_re10k/re10k_official_eval_stats/benchmark.json
outputs/olp_re10k/re10k_official_eval_stats/olp_stats_summary.json
outputs/olp_re10k/re10k_official_eval_stats/olp_stats_summary_by_overlap.json
outputs/olp_re10k/re10k_official_eval_stats/olp_stats_per_scene.csv

outputs/olp_re10k/re10k_olp_ft_eval/benchmark.json
outputs/olp_re10k/re10k_olp_ft_eval/olp_stats_summary.json
outputs/olp_re10k/re10k_olp_ft_eval/olp_stats_summary_by_overlap.json
outputs/olp_re10k/re10k_olp_ft_eval/olp_stats_per_scene.csv
```

재구성 이미지:

```text
outputs/olp_re10k/re10k_official_eval_stats/<scene>/color/*.png
outputs/olp_re10k/re10k_olp_ft_eval/<scene>/color/*.png
```

서버에서 로컬 Mac으로 가져올 때는 Mac 터미널에서 실행합니다.

```bash
rsync -avz --progress \
  -e "ssh -p <SSH_PORT>" \
  <USER>@<SERVER_HOST>:/workspace/olp_h_re10k_slot0/NoPoSplat/outputs/olp_re10k/ \
  "/Users/kwon_junyoung/Desktop/olp_re10k_outputs/"
```

## 17. 결과 해석 문구

오늘 pilot 비교:

```text
official NoPoSplat re10k.ckpt
vs
NoPoSplat + OLP-H fine-tuned
```

말할 수 있는 것:

```text
official pretrained baseline 대비 OLP-H adaptation 결과
```

말하면 안 되는 것:

```text
same-budget FT control 대비 순수 OLP-H 효과
```

그 주장은 baseline FT control을 같은 step, LR, batch, seed로 따로 돌린 뒤에만
가능합니다.
