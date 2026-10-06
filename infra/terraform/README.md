# infra/terraform — NCP 서버 자동화

ma5-bot 이 돌 네이버 클라우드(VPC) 서버 한 대를 코드로 만든다.
VPC · 서브넷 · ACG(방화벽) · 서버(KVM 3세대) · 공인 IP · `/data` 블록 스토리지, 그리고 첫 부팅 때
`/data` 마운트 → 스왑 → 패키지 → 저장소 클론 → `ma5-deploy` 설치 → venv 까지 자동으로 끝난다.
`.env` 만 사람이 넣는다.

## 준비

1. NCP 콘솔 → 마이페이지 → 인증키 관리에서 API 인증키를 만든다.
2. 노트북에 terraform 설치 (>= 1.5).
3. 이 폴더에서:

```bash
export NCLOUD_ACCESS_KEY=... NCLOUD_SECRET_KEY=...
cp terraform.tfvars.example terraform.tfvars   # ssh_public_key 를 내 공개키로
make init
make plan        # 뭘 만들지 확인. 이미지 이름이 안 맞으면 server_image_name 을 고친다.
make apply
```

## 서버 뜬 뒤

```bash
make env         # ../../.env 를 서버 /opt/ma5-bot/.env 로 (ENV_FILE=경로 로 바꿀 수 있음)
make bootstrap   # cli.py check → systemd 등록 → 시작
make status / make logs / make ssh
make deploy      # push 된 코드 반영 (BRANCH=다른브랜치 가능)
```

첫 부팅 로그는 서버의 `/var/log/ma5-init.log` 에 있다.

## 결정한 것들

- **기존 서버는 건드리지 않는다.** 지금 서버에는 다른 docker 앱이 같이 돌고 있어서 terraform 에 import 하면
  변경·반납 사고 위험이 있다. 새 서버를 따로 만들고, 봇을 옮긴 뒤 옛 서버는 콘솔에서 정리한다.
- 서버 스펙 기본값은 지금 쓰는 `s2-g3a` (vCPU 2 · 8GB). 루트 20GB + `/data` 20GB.
- 들어오는 포트는 22 뿐. `ssh_allowed_cidr` 를 집 IP 로 좁히는 걸 권장.
- 비밀번호 로그인은 끄고 `ssh_public_key` 로만 들어간다. NCP 로그인 키(`keys/*.pem`)는 root 비밀번호
  복호화용 예비 수단이고 git 에 안 올라간다.
- 상태 파일은 로컬. 여럿이 쓰게 되면 `versions.tf` 의 주석대로 Object Storage 백엔드로 옮긴다.
- 서버·블록 스토리지에 반납 보호를 켜 둔다 (`protect_termination`). `make destroy` 하려면 먼저 false 로.
- 초기화 스크립트는 첫 부팅에만 돈다. 바꿔도 떠 있는 서버는 다시 안 만든다 (`ignore_changes`).
