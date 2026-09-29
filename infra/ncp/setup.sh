#!/usr/bin/env bash
#
# infra/ncp/setup.sh — 네이버 클라우드 Micro Server 초기 설정
#
# 새로 만든 우분투 서버에 SSH 로 접속해서 실행한다.
#   curl -fsSL <이 파일 주소> | bash
# 또는 저장소를 올린 뒤:
#   bash infra/ncp/setup.sh
# 수동 매매 전용 (자동매매 없이 텔레그램 /buy /sell 만):
#   BOT_MODE=manual bash infra/ncp/setup.sh
#
set -euo pipefail

APP_USER="${SUDO_USER:-$(whoami)}"
SERVICE_NAME="ma5-bot"
BOT_MODE="${BOT_MODE:-serve}"          # serve = 자동매매 + 텔레그램, manual = 텔레그램 수동 매매만
case "$BOT_MODE" in serve|manual) ;; *) echo "BOT_MODE 는 serve 또는 manual"; exit 1 ;; esac

# 스크립트가 소스 트리 안에 있으면 그 경로를 쓴다 (어디에 올렸든 동작하도록).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_GUESS="$(cd "$_SCRIPT_DIR/../.." && pwd)"
if [ -z "${APP_DIR:-}" ] && [ -f "$_GUESS/cli.py" ]; then
  APP_DIR="$_GUESS"
else
  APP_DIR="${APP_DIR:-/opt/ma5-bot}"
fi
echo "앱 경로: $APP_DIR  (실행 모드: $BOT_MODE)"

echo "== 1. 시간대를 KST 로 =="
# 장 시간 판단이 전부 KST 기준이다. 서버가 UTC 면 로그 시각이 9시간 어긋나 헷갈린다.
sudo timedatectl set-timezone Asia/Seoul
date

echo
echo "== 2. 패키지 설치 =="
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git

PY_VER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "python3 = ${PY_VER}"
python3 - <<'PYCHECK'
import sys
if sys.version_info < (3, 10):
    sys.exit("파이썬 3.10 이상이 필요하다. deadsnakes PPA 로 3.11 을 설치해라.")
print("파이썬 버전 OK")
PYCHECK

echo
echo "== 3. 앱 디렉터리 =="
sudo mkdir -p "$APP_DIR"
sudo chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

if [ ! -d "$APP_DIR/.git" ] && [ -n "${REPO_URL:-}" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi

if [ ! -f "$APP_DIR/cli.py" ]; then
  echo
  echo "‼ $APP_DIR 에 소스가 없다. 아래 중 하나로 올린 뒤 이 스크립트를 다시 실행해라."
  echo "   - REPO_URL=https://github.com/... bash infra/ncp/setup.sh"
  echo "   - 로컬에서: scp -r ./* <user>@<서버IP>:$APP_DIR/"
  exit 1
fi

echo
echo "== 4. 가상환경 + 의존성 =="
cd "$APP_DIR"
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt
echo "설치 완료"

echo
echo "== 5. .env 확인 =="
if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
  echo "‼ .env 를 만들었다. 편집해서 한투 키와 텔레그램 토큰을 넣어라:"
  echo "    nano $APP_DIR/.env"
  echo "  VM 에서는 STATE_BACKEND=local, TOKEN_CACHE=file 로 둔다."
  echo "  넣은 뒤 이 스크립트를 다시 실행하면 서비스까지 등록된다."
  exit 0
fi
chmod 600 "$APP_DIR/.env"

echo
echo "== 6. 연결 점검 =="
./.venv/bin/python cli.py check || {
  echo "‼ 점검 실패. .env 를 고친 뒤 다시 실행해라."
  exit 1
}

echo
echo "== 7. systemd 서비스 등록 =="
sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<UNIT
[Unit]
Description=MA5 돌파 역발상 봇 (한투 OpenAPI)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=TZ=Asia/Seoul
ExecStart=${APP_DIR}/.venv/bin/python ${APP_DIR}/cli.py ${BOT_MODE}
Restart=always
RestartSec=10
StandardOutput=append:${APP_DIR}/bot.log
StandardError=append:${APP_DIR}/bot.log

[Install]
WantedBy=multi-user.target
UNIT

echo
echo "== 8. 로그 로테이션 =="
# 서비스가 bot.log 를 append 로 계속 물고 있으므로 copytruncate 가 필요하다.
# 이게 없으면 로테이션 후에도 지워진 파일에 계속 써서 디스크만 먹는다.
sudo tee "/etc/logrotate.d/${SERVICE_NAME}" >/dev/null <<ROTATE
${APP_DIR}/bot.log {
    weekly
    rotate 8
    maxsize 100M
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
ROTATE
sudo logrotate --debug "/etc/logrotate.d/${SERVICE_NAME}" >/dev/null && echo "로테이션 설정 확인 OK"

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
sleep 3
sudo systemctl status "${SERVICE_NAME}" --no-pager -l | head -20

echo
echo "════════════════════════════════════════════════"
echo "설정 완료."
echo "  상태   : sudo systemctl status ${SERVICE_NAME}"
echo "  로그   : tail -f ${APP_DIR}/bot.log"
echo "  재시작 : sudo systemctl restart ${SERVICE_NAME}"
echo "  중지   : sudo systemctl stop ${SERVICE_NAME}"
echo "════════════════════════════════════════════════"
