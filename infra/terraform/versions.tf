terraform {
  required_version = ">= 1.5"

  required_providers {
    ncloud = {
      source  = "NaverCloudPlatform/ncloud"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.5"
    }
  }

  # 상태 파일은 기본으로 로컬(terraform.tfstate)에 둔다. 사람이 둘 이상이거나
  # 노트북을 바꿔가며 쓰면 NCP Object Storage(S3 호환)로 옮긴다:
  #   backend "s3" {
  #     bucket                      = "ma5-bot-tfstate"
  #     key                         = "ncp/terraform.tfstate"
  #     region                      = "kr-standard"
  #     endpoints                   = { s3 = "https://kr.object.ncloudstorage.com" }
  #     skip_credentials_validation = true
  #     skip_region_validation      = true
  #     skip_requesting_account_id  = true
  #     skip_metadata_api_check     = true
  #     skip_s3_checksum            = true
  #     use_path_style              = true
  #   }
  # 인증은 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY 에 NCP 키를 넣어 준다.
}

provider "ncloud" {
  # NCLOUD_ACCESS_KEY / NCLOUD_SECRET_KEY / NCLOUD_REGION 환경변수로 받는다.
  region      = var.region
  site        = "public"
  support_vpc = true
}
