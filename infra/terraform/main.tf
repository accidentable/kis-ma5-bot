# ma5-bot 전용 NCP(VPC) 서버 한 대.
#
#   VPC ─ 서브넷(PUBLIC) ─ NIC(+ACG) ─ 서버(KVM 3세대) ─ 공인 IP
#                                        └ 블록 스토리지 → /data
#
# 서버는 첫 부팅 때 init.sh 를 돌려 /data 마운트·스왑·패키지·저장소 클론·
# 배포 스크립트 설치까지 끝낸다. .env 만 사람이 넣는다 (make env).

locals {
  name = var.name
}

# ---------------------------------------------------------------- 네트워크
resource "ncloud_vpc" "this" {
  name            = local.name
  ipv4_cidr_block = var.vpc_cidr
}

resource "ncloud_subnet" "public" {
  name           = "${local.name}-public"
  vpc_no         = ncloud_vpc.this.vpc_no
  subnet         = cidrsubnet(var.vpc_cidr, 8, 1) # 10.0.1.0/24
  zone           = var.zone
  network_acl_no = ncloud_vpc.this.default_network_acl_no
  subnet_type    = "PUBLIC"
  usage_type     = "GEN"
}

resource "ncloud_access_control_group" "bot" {
  name        = "${local.name}-acg"
  description = "ma5-bot: SSH only inbound"
  vpc_no      = ncloud_vpc.this.vpc_no
}

# 봇은 텔레그램 롱폴링·한투 API 호출만 하므로 들어오는 건 SSH 뿐이다.
resource "ncloud_access_control_group_rule" "bot" {
  access_control_group_no = ncloud_access_control_group.bot.id

  inbound {
    protocol    = "TCP"
    ip_block    = var.ssh_allowed_cidr
    port_range  = "22"
    description = "ssh"
  }

  outbound {
    protocol    = "TCP"
    ip_block    = "0.0.0.0/0"
    port_range  = "1-65535"
    description = "all tcp"
  }
  outbound {
    protocol    = "UDP"
    ip_block    = "0.0.0.0/0"
    port_range  = "1-65535"
    description = "all udp (dns/ntp)"
  }
  outbound {
    protocol    = "ICMP"
    ip_block    = "0.0.0.0/0"
    description = "icmp"
  }
}

resource "ncloud_network_interface" "bot" {
  name                  = "${local.name}-nic"
  subnet_no             = ncloud_subnet.public.id
  access_control_groups = [ncloud_access_control_group.bot.id]
}

# ---------------------------------------------------------------- 서버
# NCP 가 root 비밀번호를 이 키로 암호화해 준다. 평소엔 SSH 공개키로 들어가고,
# 키를 잃었을 때 콘솔에서 비밀번호를 확인하는 용도다.
resource "ncloud_login_key" "bot" {
  key_name = "${local.name}-key"
}

resource "local_sensitive_file" "login_key" {
  content         = ncloud_login_key.bot.private_key
  filename        = "${path.module}/keys/${local.name}-key.pem"
  file_permission = "0600"
}

data "ncloud_server_image_numbers" "ubuntu" {
  server_image_name = var.server_image_name
  filter {
    name   = "hypervisor_type"
    values = ["KVM"]
  }
}

data "ncloud_server_specs" "bot" {
  filter {
    name   = "server_spec_code"
    values = [var.server_spec_code]
  }
}

resource "ncloud_init_script" "bot" {
  name        = "${local.name}-init"
  description = "ma5-bot first boot: /data, swap, packages, clone, deploy script"
  os_type     = "LNX"
  content = templatefile("${path.module}/init.sh.tftpl", {
    ssh_public_key = trimspace(var.ssh_public_key)
    repo_url       = var.repo_url
    repo_branch    = var.repo_branch
    bot_mode       = var.bot_mode
    deploy_script  = file("${path.module}/../ncp/ma5-deploy")
  })
}

resource "ncloud_server" "bot" {
  name                          = local.name
  description                   = "KIS auto-trading bot (ma5-bot)"
  subnet_no                     = ncloud_subnet.public.id
  zone                          = var.zone
  server_image_number           = data.ncloud_server_image_numbers.ubuntu.image_number_list[0].server_image_number
  server_spec_code              = data.ncloud_server_specs.bot.server_spec_list[0].server_spec_code
  base_block_storage_size       = var.root_disk_gb
  login_key_name                = ncloud_login_key.bot.key_name
  init_script_no                = ncloud_init_script.bot.id
  is_protect_server_termination = var.protect_termination

  network_interface {
    network_interface_no = ncloud_network_interface.bot.id
    order                = 0
  }

  lifecycle {
    # 초기화 스크립트 내용이 바뀌어도 서버를 새로 만들지 않는다 (이미 뜬 서버엔 어차피 안 돈다).
    ignore_changes = [init_script_no]
  }
}

resource "ncloud_public_ip" "bot" {
  server_instance_no = ncloud_server.bot.id
  description        = "${local.name} public ip"
}

# /data — 백테스트 데이터·스왑. 루트 디스크와 분리해 두면 서버를 갈아끼워도 데이터는 남는다.
resource "ncloud_block_storage" "data" {
  name               = "${local.name}-data"
  server_instance_no = ncloud_server.bot.id
  size               = var.data_disk_gb
  zone               = var.zone
  hypervisor_type    = "KVM"
  volume_type        = "CB1"
  return_protection  = var.protect_termination
}
