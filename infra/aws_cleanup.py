"""
infra/aws_cleanup.py — 예전 AWS 스택 정리

네이버 클라우드로 옮기면 시드니 리전에 남은 옛 리소스가 요금만 먹는다.
기본 동작은 **조회만** 한다. 실제 삭제는 --yes 를 붙여야 실행된다.

  python infra/aws_cleanup.py                                  # 무엇이 남았는지 보기
  python infra/aws_cleanup.py --region ap-southeast-2          # 리전 지정
  python infra/aws_cleanup.py --prefix stock-bot --yes         # 실제 삭제

삭제는 되돌릴 수 없다. 목록을 먼저 확인하고 실행해라.
"""
from __future__ import annotations

import argparse
import sys

import boto3
from botocore.exceptions import ClientError


def _collect(region: str, prefix: str) -> dict[str, list[str]]:
    s = boto3.Session(region_name=region)
    found: dict[str, list[str]] = {}

    def _try(label: str, fn):
        try:
            items = fn()
            if items:
                found[label] = items
        except ClientError as e:
            print(f"  ({label} 조회 건너뜀: {e.response['Error']['Code']})")
        except Exception as e:
            print(f"  ({label} 조회 건너뜀: {type(e).__name__})")

    lam = s.client("lambda")
    _try("Lambda 함수", lambda: [
        f["FunctionName"] for p in lam.get_paginator("list_functions").paginate()
        for f in p["Functions"] if f["FunctionName"].startswith(prefix)
    ])

    apigw = s.client("apigatewayv2")
    _try("API Gateway", lambda: [
        f"{a['ApiId']} ({a['Name']})" for a in apigw.get_apis()["Items"]
        if a["Name"].startswith(prefix)
    ])

    events = s.client("events")
    _try("EventBridge 규칙", lambda: [
        r["Name"] for r in events.list_rules(NamePrefix=prefix)["Rules"]
    ])

    sched = s.client("scheduler")
    _try("EventBridge 스케줄", lambda: [
        x["Name"] for x in sched.list_schedules(NamePrefix=prefix)["Schedules"]
    ])

    ddb = s.client("dynamodb")
    _try("DynamoDB 테이블", lambda: [
        t for t in ddb.list_tables()["TableNames"] if t.startswith(prefix)
    ])

    ssm = s.client("ssm")
    _try("SSM 파라미터", lambda: [
        p["Name"] for p in ssm.describe_parameters(
            ParameterFilters=[{"Key": "Name", "Option": "BeginsWith", "Values": [f"/{prefix}"]}]
        )["Parameters"]
    ])

    s3 = s.client("s3")
    _try("S3 버킷", lambda: [
        b["Name"] for b in s3.list_buckets()["Buckets"] if b["Name"].startswith(prefix)
    ])

    iam = s.client("iam")
    _try("IAM 역할", lambda: [
        r["RoleName"] for p in iam.get_paginator("list_roles").paginate()
        for r in p["Roles"] if r["RoleName"].startswith(prefix)
    ])

    return found


def _delete(region: str, prefix: str, found: dict[str, list[str]]) -> None:
    s = boto3.Session(region_name=region)

    for name in found.get("EventBridge 스케줄", []):
        s.client("scheduler").delete_schedule(Name=name)
        print(f"  삭제: 스케줄 {name}")

    for name in found.get("EventBridge 규칙", []):
        ev = s.client("events")
        targets = ev.list_targets_by_rule(Rule=name)["Targets"]
        if targets:
            ev.remove_targets(Rule=name, Ids=[t["Id"] for t in targets])
        ev.delete_rule(Name=name)
        print(f"  삭제: 규칙 {name}")

    for entry in found.get("API Gateway", []):
        api_id = entry.split()[0]
        s.client("apigatewayv2").delete_api(ApiId=api_id)
        print(f"  삭제: API {entry}")

    for name in found.get("Lambda 함수", []):
        s.client("lambda").delete_function(FunctionName=name)
        print(f"  삭제: 람다 {name}")

    for name in found.get("DynamoDB 테이블", []):
        s.client("dynamodb").delete_table(TableName=name)
        print(f"  삭제: 테이블 {name}")

    for name in found.get("SSM 파라미터", []):
        s.client("ssm").delete_parameter(Name=name)
        print(f"  삭제: 파라미터 {name}")

    for name in found.get("S3 버킷", []):
        bucket = s.resource("s3").Bucket(name)
        bucket.objects.all().delete()
        bucket.delete()
        print(f"  삭제: 버킷 {name}")

    iam = s.client("iam")
    for name in found.get("IAM 역할", []):
        for p in iam.list_attached_role_policies(RoleName=name)["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=name, PolicyArn=p["PolicyArn"])
        for p in iam.list_role_policies(RoleName=name)["PolicyNames"]:
            iam.delete_role_policy(RoleName=name, PolicyName=p)
        iam.delete_role(RoleName=name)
        print(f"  삭제: 역할 {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="옛 AWS 스택 조회/정리")
    ap.add_argument("--region", default="ap-southeast-2", help="기본: ap-southeast-2 (시드니)")
    ap.add_argument("--prefix", default="stock-bot", help="리소스 이름 접두어. 기본: stock-bot")
    ap.add_argument("--yes", action="store_true", help="실제로 삭제한다 (없으면 조회만)")
    args = ap.parse_args()

    print(f"리전 {args.region} / 접두어 '{args.prefix}' 조회 중...\n")
    found = _collect(args.region, args.prefix)

    if not found:
        print("해당하는 리소스가 없다.")
        return 0

    total = 0
    for label, items in found.items():
        print(f"[{label}] {len(items)}개")
        for i in items:
            print(f"   - {i}")
        total += len(items)
    print(f"\n합계 {total}개")

    if not args.yes:
        print("\n조회만 했다. 실제로 지우려면 --yes 를 붙여라.")
        print("삭제는 되돌릴 수 없으니 위 목록을 꼭 확인해라.")
        return 0

    print(f"\n‼ {total}개를 삭제한다. 계속하려면 'delete' 를 입력해라: ", end="")
    if input().strip().lower() != "delete":
        print("취소했다.")
        return 1

    print()
    _delete(args.region, args.prefix, found)
    print("\n정리 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
