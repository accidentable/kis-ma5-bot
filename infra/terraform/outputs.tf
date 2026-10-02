output "public_ip" {
  value = ncloud_public_ip.bot.public_ip
}

output "private_ip" {
  value = ncloud_server.bot.private_ip
}

output "server_instance_no" {
  value = ncloud_server.bot.id
}

output "ssh" {
  value = "ssh root@${ncloud_public_ip.bot.public_ip}"
}

output "login_key_file" {
  description = "NCP 로그인 키(root 비밀번호 복호화용). SSH 키가 아니다."
  value       = local_sensitive_file.login_key.filename
}
