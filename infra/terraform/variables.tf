variable "region" {
  description = "NCP 리전 코드"
  type        = string
  default     = "KR"
}

variable "zone" {
  description = "가용 영역. 서브넷·서버·블록 스토리지가 전부 같은 존이어야 한다."
  type        = string
  default     = "KR-2"
}

variable "name" {
  description = "리소스 이름 접두어 (영문 소문자·숫자·하이픈)"
  type        = string
  default     = "ma5-bot"
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "server_image_name" {
  description = "KVM 3세대 서버 이미지 이름. terraform plan 이 못 찾으면 NCP 콘솔의 서버 이미지 목록에서 이름을 확인한다."
  type        = string
  default     = "ubuntu-24.04-base"
}

variable "server_spec_code" {
  description = "서버 스펙 코드. 지금 쓰는 서버가 s2-g3a (vCPU 2, 8GB) 다."
  type        = string
  default     = "s2-g3a"
}

variable "root_disk_gb" {
  description = "루트 디스크 크기(GB). 10GB 는 금방 찬다."
  type        = number
  default     = 20
}

variable "data_disk_gb" {
  description = "/data 로 붙일 블록 스토리지 크기(GB, 10 단위). 백테스트 데이터·스왑이 여기 들어간다."
  type        = number
  default     = 20
}

variable "ssh_public_key" {
  description = "root 의 authorized_keys 에 넣을 SSH 공개키 한 줄 (ssh-ed25519 ...). 비밀번호 로그인은 막는다."
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "22 번 포트를 열어 줄 CIDR. 집/사무실 IP 로 좁혀라."
  type        = string
  default     = "0.0.0.0/0"
}

variable "repo_url" {
  type    = string
  default = "https://github.com/accidentable/kis-ma5-bot.git"
}

variable "repo_branch" {
  description = "서버에 올릴 브랜치. ma5-deploy 의 기본값으로도 쓰인다."
  type        = string
  default     = "backtest-universe-iofw7v"
}

variable "bot_mode" {
  description = "systemd 가 실행할 cli.py 서브커맨드: serve(자동매매+텔레그램) | manual(텔레그램 수동매매만)"
  type        = string
  default     = "serve"
  validation {
    condition     = contains(["serve", "manual"], var.bot_mode)
    error_message = "bot_mode 는 serve 또는 manual."
  }
}

variable "protect_termination" {
  description = "콘솔·API 실수로 서버가 반납되지 않게 보호"
  type        = bool
  default     = true
}
