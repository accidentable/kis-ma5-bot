# Terraform 배우기 — ma5-bot 서버를 직접 코드로 띄우기

이 문서는 네가 **직접 손으로** 따라 하는 교재다. 이 폴더의 `.tf` 파일들은 완성본이 아니라 읽을거리이고,
각 단계 끝에 "직접 해보기"가 있다. 명령은 네 노트북(또는 이 서버의 `claude` 계정)에서 네가 친다.

---

## 0. Terraform 이 뭔가

**Terraform** 은 "서버·네트워크·디스크 같은 클라우드 자원을 글로 적어 두면, 그 글대로 만들어 주는 도구"다.
HashiCorp 라는 회사가 만들었고, 무료다.

비유: 가구 조립 설명서. 설명서(코드)에 "다리 4개, 상판 1개, 나사 8개"라고 적어 두면,
Terraform 이 창고(클라우드)에 전화해서 그대로 가져다 조립한다. 다음에 "다리 6개"로 고치면 **2개만 더** 가져온다.

지금까지는 네이버 클라우드 콘솔(웹 화면)에서 마우스로 서버를 만들고, `setup.sh` 를 직접 돌렸다.
그 방식의 문제:
- 어떻게 만들었는지 기록이 없다. 서버가 죽으면 기억을 더듬어 다시 눌러야 한다.
- "방화벽에 22번만 열었나?" 같은 걸 확인하려면 화면을 뒤져야 한다.

Terraform 으로 적어 두면 **서버 구성 = 저장소에 들어 있는 텍스트** 가 되고, git 으로 이력이 남는다.

### 핵심 단어 6개 (이것만 알면 된다)

| 단어 | 뜻 | 비유 |
|---|---|---|
| **Provider** | 특정 클라우드와 대화하는 플러그인. 우리는 네이버 클라우드용 `ncloud` | 창고마다 다른 전화번호 |
| **Resource** | 만들 물건 하나 (서버 1대, 디스크 1개, 방화벽 규칙 1벌) | 설명서의 부품 한 줄 |
| **Data source** | 만들진 않고 **조회만** 하는 것 (예: "우분투 24.04 이미지 번호가 뭐지?") | 창고 카탈로그 찾아보기 |
| **Variable** | 바꿔 끼울 값 (서버 크기, SSH 키) | 설명서의 빈칸 |
| **Output** | 다 만든 뒤 알려줄 값 (공인 IP) | 완성 후 "여기 있어요" 메모 |
| **State** | Terraform 이 "내가 뭘 만들었는지" 적어 두는 장부 파일 `terraform.tfstate` | 창고 반출 기록 |

### 명령은 4개

```
terraform init      # 플러그인(provider) 내려받기. 폴더당 처음 한 번.
terraform plan      # "이렇게 바꿀 건데 괜찮아?" 미리보기. 아무것도 안 만든다.
terraform apply     # 진짜 만든다 (plan 을 보여주고 yes 를 물어본다).
terraform destroy   # 장부에 적힌 걸 전부 지운다. 위험.
```

> **직접 해보기 0**
> 이 서버의 `claude` 계정엔 terraform 이 `~/.local/bin/terraform` 에 깔려 있다.
> ```bash
> export PATH=$HOME/.local/bin:$PATH
> terraform version
> ```
> 노트북에서 하려면 https://developer.hashicorp.com/terraform/install 에서 받는다.

---

## 1. 파일 구조 읽기

Terraform 은 **한 폴더 안의 `.tf` 파일을 전부 합쳐서** 하나로 본다. 파일 이름은 사람 편의용이고,
순서도 상관없다 (의존 관계는 Terraform 이 알아서 푼다). 그래서 관례로 이렇게 나눈다:

```
infra/terraform/
├── versions.tf      어떤 provider 를 몇 버전으로 쓸지, 장부(state)를 어디 둘지
├── variables.tf     빈칸 목록
├── main.tf          실제로 만들 것들 (resource, data)
├── outputs.tf       끝나고 보여줄 값
├── init.sh.tftpl    서버가 처음 켜질 때 돌릴 셸 스크립트 (템플릿)
├── terraform.tfvars.example   빈칸을 채운 예시. 복사해서 terraform.tfvars 로 쓴다 (git 제외)
├── Makefile         자주 치는 명령 묶음
└── .terraform.lock.hcl   init 이 고른 provider 버전을 고정 (git 에 넣는다)
```

읽는 순서: `versions.tf` → `variables.tf` → `main.tf` → `outputs.tf` → `init.sh.tftpl`.

---

## 2. versions.tf — 어떤 플러그인을 쓰나

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    ncloud = {
      source  = "NaverCloudPlatform/ncloud"
      version = "~> 4.0"
    }
  }
}
provider "ncloud" {
  region      = var.region
  site        = "public"
  support_vpc = true
}
```

- `required_providers`: "ncloud 라는 이름으로 NaverCloudPlatform 이 만든 플러그인 4.x 를 쓰겠다".
  `~> 4.0` 은 "4.0 이상 5.0 미만". `terraform init` 이 이걸 보고 registry.terraform.io 에서 내려받는다.
- `provider "ncloud" { ... }`: 플러그인 설정. 인증키는 여기 적지 않고 **환경변수**
  `NCLOUD_ACCESS_KEY` / `NCLOUD_SECRET_KEY` 로 받는다. 코드에 비밀을 넣으면 git 에 올라가기 때문이다.
- `support_vpc = true`: 네이버 클라우드는 옛날 방식(Classic)과 새 방식(VPC)이 있는데, 지금 서버는 VPC 다.
  **VPC** 는 "내 전용 가상 네트워크" — 아파트 한 동을 통째로 빌린 것. 그 안에 방(서브넷)을 나눈다.
- 주석 처리된 `backend "s3"`: 장부(state)를 로컬 파일 대신 네이버 Object Storage 에 두는 설정. 혼자 쓰는 동안은 필요 없다.

> **직접 해보기 1**
> ```bash
> cd ~/kis-lab/infra/terraform
> terraform init
> ls .terraform/providers/          # 내려받은 플러그인이 보인다
> cat .terraform.lock.hcl | head    # 어떤 버전을 골랐는지 고정해 둔 파일
> ```

---

## 3. variables.tf — 빈칸

```hcl
variable "server_spec_code" {
  description = "서버 스펙 코드"
  type        = string
  default     = "s2-g3a"
}
variable "ssh_public_key" {
  type = string          # default 가 없다 → 반드시 채워야 한다
}
```

- `default` 가 있으면 안 채워도 되고, 없으면 `terraform.tfvars` 파일이나 `-var` 옵션으로 줘야 한다.
- 다른 파일에서는 `var.server_spec_code` 처럼 쓴다.
- `validation` 블록은 잘못된 값을 plan 단계에서 막는다 (`bot_mode` 참고).

왜 `s2-g3a` 인가: 지금 서버가 그 스펙이다 (`cat /sys/class/dmi/id/product_name` 으로 확인했다).
`s2` = vCPU 2개, `g3` = 3세대(KVM 가상화), `a` 는 세부 라인.

> **직접 해보기 2**
> ```bash
> cp terraform.tfvars.example terraform.tfvars
> cat ~/.ssh/id_ed25519.pub        # 없으면 ssh-keygen -t ed25519 로 만든다
> ```
> `terraform.tfvars` 의 `ssh_public_key` 에 그 한 줄을 붙여 넣는다. `ssh_allowed_cidr` 는 `curl ifconfig.me` 로
> 나온 네 IP 뒤에 `/32` 를 붙이면 "내 집에서만 SSH 허용"이 된다.

---

## 4. main.tf — 진짜 만들 것들

위에서 아래로 **네트워크 → 서버 → 디스크** 순이다. 각 블록의 모양은 늘 같다:

```hcl
resource "<종류>" "<내가 붙인 별명>" {
  인자 = 값
}
```

다른 블록에서 참조할 땐 `<종류>.<별명>.<속성>` 이다. 예: `ncloud_vpc.this.vpc_no`.
**이 참조가 곧 의존 관계**다. 서브넷이 `ncloud_vpc.this.vpc_no` 를 쓰면 Terraform 은 VPC 를 먼저 만든다.

### 4-1. 네트워크 4개

| 블록 | 뭘 만드나 | 비유 |
|---|---|---|
| `ncloud_vpc` | 사설망 `10.0.0.0/16` (IP 65,536개짜리 내 전용 구역) | 아파트 한 동 |
| `ncloud_subnet` | 그 안의 `10.0.1.0/24` (IP 256개), `PUBLIC` = 인터넷에 닿는 방 | 층 하나 |
| `ncloud_access_control_group` + `_rule` | 방화벽. **ACG** 라고 부른다 | 현관 출입 규칙 |
| `ncloud_network_interface` | 서버에 꽂을 랜카드. ACG 는 서버가 아니라 **랜카드에** 붙는다 | 현관문 |

`cidrsubnet(var.vpc_cidr, 8, 1)` 은 "10.0.0.0/16 을 8비트 더 쪼갠 것 중 1번째" = `10.0.1.0/24`.
지금 서버 IP 가 `10.0.1.6` 인 것과 같은 구조로 맞췄다.

방화벽 규칙을 보면 **들어오는(inbound)** 건 22번(SSH)뿐이다. 봇은 텔레그램과 한투 서버에
**먼저 말을 거는** 쪽이라서 밖에서 들어올 포트가 필요 없다. 나가는(outbound) 건 전부 연다.

### 4-2. 서버

```hcl
data "ncloud_server_image_numbers" "ubuntu" {
  server_image_name = var.server_image_name     # "ubuntu-24.04-base"
  filter { name = "hypervisor_type"  values = ["KVM"] }
}
```
`data` 는 조회다. "우분투 24.04 KVM 이미지의 **번호**" 를 네이버에 물어본다. 서버를 만들 땐 이름이 아니라
번호를 줘야 하는데, 번호는 외울 수 없으니 코드가 대신 찾게 한다.
(`terraform plan` 을 돌렸을 때 "image_number_list is empty" 같은 오류가 나면 이름이 틀린 거다.
`output_file = "image.json"` 을 임시로 넣고 plan 하면 전체 목록이 파일로 떨어진다.)

```hcl
resource "ncloud_server" "bot" {
  subnet_no           = ncloud_subnet.public.id
  server_image_number = data.ncloud_server_image_numbers.ubuntu.image_number_list[0].server_image_number
  server_spec_code    = data.ncloud_server_specs.bot.server_spec_list[0].server_spec_code
  init_script_no      = ncloud_init_script.bot.id
  is_protect_server_termination = true
  network_interface { network_interface_no = ncloud_network_interface.bot.id  order = 0 }
  lifecycle { ignore_changes = [init_script_no] }
}
```
- `init_script_no`: 서버가 **처음 켜질 때 딱 한 번** 돌릴 스크립트. 5장에서 본다.
- `is_protect_server_termination = true`: 콘솔에서 실수로 "반납" 눌러도 안 지워진다.
- `lifecycle.ignore_changes`: 나중에 init 스크립트를 고쳐도 "서버 다시 만들어야 해" 라고 하지 말라는 뜻.
  Terraform 은 기본적으로 **바꿀 수 없는 속성이 바뀌면 지우고 새로 만든다**. 이걸 모르고 apply 하면
  서버가 날아간다. plan 출력에서 `-/+` (destroy and then create replacement) 표시를 항상 확인해라.

`ncloud_login_key`: 네이버가 root 비밀번호를 이 키로 암호화해서 보관한다. SSH 키가 **아니다**.
우리는 SSH 는 공개키로 들어가고(init 스크립트가 넣음), 이 키는 비밀번호를 잃었을 때용으로만 `keys/` 에 저장한다 (git 제외).

### 4-3. 공인 IP 와 /data 디스크

```hcl
resource "ncloud_public_ip" "bot"     { server_instance_no = ncloud_server.bot.id }
resource "ncloud_block_storage" "data" {
  server_instance_no = ncloud_server.bot.id
  size = 20   zone = var.zone   hypervisor_type = "KVM"   volume_type = "CB1"
}
```
- VPC 의 PUBLIC 서브넷이어도 **공인 IP 는 따로 신청**해야 밖에서 들어온다.
- 블록 스토리지 = 외장하드. 루트 디스크(OS)와 분리해 두면, 서버를 새로 만들어도 데이터만 떼어 옮길 수 있다.
  지금 서버의 `/data` 20GB 와 같은 구성이다. `CB1` 은 네이버의 일반 SSD 등급.

> **직접 해보기 3** — 아직 키 없이도 된다
> ```bash
> terraform validate      # 문법·참조 검사
> terraform graph | head  # 의존 관계를 그래프(dot 형식)로. 어떤 게 먼저 만들어지는지 보인다
> ```

---

## 5. init.sh.tftpl — 서버가 처음 켜질 때

`.tftpl` 은 **템플릿**이다. `${ssh_public_key}` 같은 자리에 main.tf 의 `templatefile(...)` 이 값을 끼워 넣어
완성된 셸 스크립트를 만들고, 그걸 `ncloud_init_script` 로 네이버에 올린다. 서버는 첫 부팅에 root 로 이걸 돌린다.

하는 일 순서 (지금 서버에서 손으로 했던 일들을 전부 적은 것):
1. `/root/.ssh/authorized_keys` 에 네 공개키 넣고, 비밀번호 로그인 끄기
2. 시간대 KST (장 시간 판단이 전부 KST 기준이라서)
3. 두 번째 디스크를 찾아 ext4 로 포맷 → `/data` 마운트 → `/etc/fstab` 등록 (재부팅해도 유지)
4. `/data/swapfile` 4GB (메모리 8GB 로 백테스트 돌릴 때 버티게)
5. python3·git 설치
6. 저장소 클론 → `/opt/ma5-bot`, `data/lab → /data/lab` 심볼릭 링크
7. `ma5-deploy` 설치 (`infra/ncp/ma5-deploy` 내용을 그대로 박아 넣는다)
8. `infra/ncp/setup.sh` 실행 → venv 까지 만들고 `.env` 가 없어서 멈춘다

**왜 .env 는 안 넣나**: init 스크립트 내용은 네이버 서버와 state 장부에 평문으로 남는다. 한투 키가 거기 있으면 안 된다.
그래서 `.env` 만 `make env`(scp)로 손으로 올린다.

주의: 템플릿 안에서 `${...}` 는 Terraform 변수다. 셸의 `${HOME}` 같은 걸 쓰려면 `$${HOME}` 으로 써야 한다.
(`$(...)` 는 셸 그대로라 괜찮다.)

> **직접 해보기 4** — 렌더링 결과 눈으로 보기
> ```bash
> terraform console
> > templatefile("init.sh.tftpl", { ssh_public_key="ssh-ed25519 TEST", repo_url="x", repo_branch="b", bot_mode="serve", deploy_script="echo deploy" })
> ```
> 완성된 스크립트가 출력된다. `exit` 로 나온다.

---

## 6. outputs.tf 와 Makefile

`output "public_ip"` 처럼 적어 두면 apply 끝에 화면에 찍히고, `terraform output -raw public_ip` 로 꺼낼 수 있다.
Makefile 은 그걸 받아서 `ssh root@<IP>` 같은 명령을 `make ssh` 한 줄로 줄인 것뿐이다. 마법은 없다.
`cat Makefile` 로 각 타깃이 어떤 명령인지 꼭 봐라.

---

## 7. 실제로 만들기 — 순서

**이 단계부터 돈이 든다** (s2-g3a 시간당 과금). 끝나면 destroy 하거나 옛 서버를 반납해라.

1. 네이버 클라우드 콘솔 → 마이페이지 → **인증키 관리** → API 인증키 생성. Access Key / Secret Key 둘 다 복사.
2. ```bash
   export NCLOUD_ACCESS_KEY=... NCLOUD_SECRET_KEY=...
   cd ~/kis-lab/infra/terraform
   terraform plan -out=tf.plan
   ```
   출력을 **끝까지** 읽는다. `Plan: 11 to add, 0 to change, 0 to destroy.` 처럼 끝나야 한다.
   `destroy` 가 0이 아니면 멈추고 이유를 찾는다.
3. `terraform apply tf.plan` → 3~5분. 끝에 `public_ip` 가 찍힌다.
4. `ssh root@<IP> tail -f /var/log/ma5-init.log` 로 init 스크립트가 끝났는지 본다 (`init done` 줄).
5. `make env` → `make bootstrap` → `make status`. 봇이 뜨면 `make logs`.
6. 옛 서버의 봇을 끄고(`sudo systemctl stop ma5-bot`), 옛 서버 `.env`·`/data/lab` 중 필요한 걸 옮긴다.

### 꼭 알아둘 것
- `terraform.tfstate` 는 **장부**다. 지우면 Terraform 은 서버가 있는지 모르게 되고, 다음 apply 때 **또 만든다**.
  git 에는 안 올리지만 (키·IP 가 들어 있다) 백업은 해 둬라.
- 콘솔에서 손으로 바꾸면 장부와 어긋난다. `terraform plan` 이 그 차이를 보여주고, apply 하면 **코드대로 되돌린다**.
  그래서 한 번 Terraform 으로 만들면 그 뒤 변경도 코드로 한다.
- 지금 쓰는 서버를 Terraform 에 넣는 건(`terraform import`) 일부러 안 했다. 거기엔 다른 docker 앱이 같이 살아서,
  실수 한 번에 그 앱까지 날아간다. 새 서버를 만들고 옮기는 게 안전하다.

---

## 8. 더 배우려면 (순서대로)

1. 공식 입문: https://developer.hashicorp.com/terraform/tutorials — "Get Started" 아무 클라우드나 하나.
2. 언어 문법(HCL): https://developer.hashicorp.com/terraform/language — `for_each`, `count`, `locals` 까지.
3. 네이버 provider 문서: https://registry.terraform.io/providers/NaverCloudPlatform/ncloud/latest/docs
   (화면이 안 뜨면 GitHub 의 `docs/` 폴더를 본다: https://github.com/NaverCloudPlatform/terraform-provider-ncloud/tree/main/docs)
4. 다음 연습 과제: 이 코드를 **module** 로 감싸서 `mock` 용 서버와 `real` 용 서버 두 대를 변수만 바꿔 만들어 보기.

---

## 9. 실전 과제: 노트북 IP 가 바뀔 때마다 SSH 허용 IP 를 코드로 갱신하기

목표: **지금 쓰는 서버**의 ACG 에서 22번 포트 허용 IP 를, 콘솔 대신 `terraform apply` 한 번으로 "지금 내 노트북 IP" 로 바꾼다.
서버 전체를 Terraform 에 넣는 게 아니라 **방화벽 규칙 하나만** 가져온다. 그래서 7장과는 **별도 폴더, 별도 장부(state)** 로 한다.

### 9-0. 원리

- `ncloud_access_control_group_rule` 리소스는 "ACG 하나에 달린 규칙 **전체**" 다. 리소스 id = ACG 번호.
- 이미 콘솔에서 만든 걸 Terraform 장부에 올리는 걸 **import** 라고 한다. 가져온 뒤엔 코드가 진실이 된다.
  → **코드에 안 적힌 규칙은 apply 때 지워진다.** 그래서 지금 ACG 에 있는 규칙을 빠짐없이 옮겨 적어야 한다.
- 내 IP 는 `http` 라는 provider 로 `https://checkip.amazonaws.com` 에 물어본다. 사람이 IP 를 안 쳐도 된다.

> 조심: 규칙을 잘못 적으면 **SSH 가 막힌다.** 그래도 콘솔에서 ACG 를 고치면 다시 들어갈 수 있으니 치명적이진 않다.
> 같은 서버의 docker 앱이 쓰는 포트(80, 443 등)가 ACG 에 열려 있다면 그것도 반드시 옮겨 적어야 한다.

### 9-1. 준비 (콘솔에서 두 가지 확인)

콘솔 → VPC → **ACG** 메뉴 → 서버에 붙은 ACG 를 연다.
1. **ACG 번호** (ACG ID, 숫자) 를 적는다.
2. **규칙 목록** 전부를 적는다: Inbound 탭과 Outbound 탭 각각 (프로토콜 / 접근소스 / 포트 / 메모).
   어느 ACG 인지 모르겠으면 콘솔 → Server → 서버 상세 → 네트워크 인터페이스 → ACG 이름을 본다.

### 9-2. 폴더 만들기 (네가 직접 친다)

```bash
mkdir -p ~/kis-lab/infra/terraform/ssh-access && cd $_
```

`versions.tf`:
```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    ncloud = { source = "NaverCloudPlatform/ncloud", version = "~> 4.0" }
    http   = { source = "hashicorp/http", version = "~> 3.4" }
  }
}
provider "ncloud" {
  region      = "KR"
  site        = "public"
  support_vpc = true
}
```

`variables.tf`:
```hcl
variable "acg_no" {
  description = "콘솔에서 본 ACG 번호"
  type        = string
}
```

`main.tf`:
```hcl
# 지금 내 공인 IP. 응답은 "1.2.3.4\n" 이라서 trimspace 로 줄바꿈을 뗀다.
data "http" "my_ip" {
  url = "https://checkip.amazonaws.com"
}

locals {
  my_cidr = "${trimspace(data.http.my_ip.response_body)}/32"
}

# 기존 규칙 뭉치를 장부에 올린다. 한 번 import 되면 이 블록은 지워도 된다 (남겨 둬도 무해).
import {
  to = ncloud_access_control_group_rule.main
  id = var.acg_no
}

resource "ncloud_access_control_group_rule" "main" {
  access_control_group_no = var.acg_no

  inbound {
    protocol    = "TCP"
    ip_block    = local.my_cidr
    port_range  = "22"
    description = "ssh from laptop (terraform)"
  }

  # ↓ 9-1 에서 적어 온 나머지 규칙을 전부 여기 옮긴다. 예:
  # inbound { protocol = "TCP"  ip_block = "0.0.0.0/0"  port_range = "443"  description = "docker app" }

  outbound {
    protocol    = "TCP"
    ip_block    = "0.0.0.0/0"
    port_range  = "1-65535"
  }
  outbound {
    protocol    = "UDP"
    ip_block    = "0.0.0.0/0"
    port_range  = "1-65535"
  }
  outbound {
    protocol = "ICMP"
    ip_block = "0.0.0.0/0"
  }
}

output "allowed_ssh_from" { value = local.my_cidr }
```

`terraform.tfvars` (git 제외):
```hcl
acg_no = "123456"
```

`.gitignore` 는 상위 폴더 것이 그대로 적용된다 (state, tfvars 제외됨).

### 9-3. 처음 한 번: import

```bash
export NCLOUD_ACCESS_KEY=... NCLOUD_SECRET_KEY=...
terraform init
terraform plan
```

plan 출력을 읽는 법:
- `ncloud_access_control_group_rule.main will be imported` — 장부에 올리겠다는 뜻. 정상.
- 그 아래 `~ update in-place` 와 `inbound` 의 `-`/`+` 줄들 — **콘솔에 있는 규칙과 코드의 차이**다.
  `-` 로 빠지는 줄이 "docker 앱 포트" 같은 거면 코드에 빠뜨린 것이니 추가하고 다시 plan.
  `-` 가 "예전 노트북 IP 의 22번" 하나뿐이고 `+` 가 지금 IP 면 정답.
- `Plan: 1 to import, 0 to add, 1 to change, 0 to destroy.` 로 끝나야 한다. `destroy` 가 1 이면 멈춘다.

맞으면 `terraform apply`. 끝나고 `ssh root@<서버IP>` 가 되는지 확인.

### 9-4. 이후 매번 (카페·회사 등 IP 가 바뀌었을 때)

```bash
cd ~/kis-lab/infra/terraform/ssh-access
terraform apply
```
plan 에 `22` 규칙의 ip_block 만 `~` 로 바뀌면 `yes`. 10초면 끝난다.
노트북 셸에 별칭을 걸어 두면 더 편하다:
```bash
alias ncp-ssh-open='cd ~/kis-lab/infra/terraform/ssh-access && terraform apply -auto-approve && cd -'
```

### 9-5. 더 해보기

- `terraform state show ncloud_access_control_group_rule.main` — 장부에 뭐가 적혔는지 본다.
- 7장의 새 서버 코드에서는 이 "내 IP 자동 조회" 를 `ssh_allowed_cidr` 변수 대신 쓰도록 바꿔 보기.
  (힌트: `data "http"` + `locals` 를 `main.tf` 로 옮기고 `var.ssh_allowed_cidr` 자리에 `local.my_cidr`.)
- IP 를 두 개 허용(집 + 회사)하려면 `inbound` 블록을 두 개 쓰거나, `dynamic "inbound"` + `for_each` 로 목록을 돌린다.
  `dynamic` 은 HCL 문법 문서의 "Dynamic Blocks" 항목.
