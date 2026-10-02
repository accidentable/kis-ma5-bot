# ma5-bot NCP 서버 — terraform 래퍼
#   필요: terraform >= 1.5, 환경변수 NCLOUD_ACCESS_KEY / NCLOUD_SECRET_KEY, terraform.tfvars
TF ?= terraform
ENV_FILE ?= ../../.env
IP = $(shell $(TF) output -raw public_ip 2>/dev/null)

.PHONY: init fmt validate plan apply destroy output ssh env bootstrap deploy logs status

init:      ; $(TF) init
fmt:       ; $(TF) fmt -recursive
validate:  ; $(TF) validate
plan:      ; $(TF) plan -out=tf.plan
apply:     ; $(TF) apply tf.plan
destroy:   ; $(TF) destroy
output:    ; $(TF) output

ssh:       ; ssh root@$(IP)

# .env 를 서버에 올린다 (git 에 없는 유일한 파일).
env:
	@test -f $(ENV_FILE) || { echo "$(ENV_FILE) 없음"; exit 1; }
	scp $(ENV_FILE) root@$(IP):/opt/ma5-bot/.env
	ssh root@$(IP) 'chmod 600 /opt/ma5-bot/.env'

# .env 를 넣은 뒤 한 번: 연결 점검 + systemd 등록 + 시작
bootstrap:
	ssh root@$(IP) 'cd /opt/ma5-bot && APP_DIR=/opt/ma5-bot bash infra/ncp/setup.sh'

# 이후 코드 반영은 이걸로 (push 된 브랜치 기준)
deploy:
	ssh root@$(IP) 'ma5-deploy $(BRANCH)'

logs:      ; ssh root@$(IP) 'tail -n 50 /opt/ma5-bot/bot.log'
status:    ; ssh root@$(IP) 'systemctl status ma5-bot --no-pager -l | head -20'
