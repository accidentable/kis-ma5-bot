"""
infra/deploy.py — AWS 배포

  python infra/deploy.py bootstrap   DynamoDB 테이블 + IAM 역할 생성
  python infra/deploy.py package     배포 zip 생성 (로컬 확인용)
  python infra/deploy.py lambdas     람다 4개 생성/갱신
  python infra/deploy.py schedules   EventBridge 스케줄 등록 (KST)
  python infra/deploy.py webhook     API Gateway + 텔레그램 웹훅 연결
  python infra/deploy.py all         위 전부 순서대로

실행 전 .env 가 채워져 있어야 한다. 환경변수는 람다 설정으로 그대로 넘어간다.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from botocore.exceptions import ClientError

import config

PROJECT = "ma5-bot"
ROLE_NAME = f"{PROJECT}-lambda-role"
SCHED_ROLE_NAME = f"{PROJECT}-scheduler-role"
RUNTIME = "python3.11"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (람다 이름 접미사, 핸들러, 타임아웃초, 메모리MB)
FUNCTIONS = [
    ("prep", "handlers.aws.prep_handler", 300, 512),
    ("premarket", "handlers.aws.premarket_handler", 180, 256),
    ("entry", "handlers.aws.entry_handler", 300, 512),
    ("monitor", "handlers.aws.monitor_handler", 120, 256),
    ("close", "handlers.aws.close_handler", 180, 256),
    ("webhook", "handlers.aws.webhook_handler", 120, 512),
]

# EventBridge Scheduler — 타임존을 직접 지정할 수 있어 UTC 변환이 필요 없다
SCHEDULES = [
    ("prep", "cron(40 8 ? * MON-FRI *)", "08:40 준비"),
    ("premarket", "cron(50 8 ? * MON-FRI *)", "08:50 프리마켓 분할진입"),
    ("entry", "cron(5 9 ? * MON-FRI *)", "09:05 진입"),
    ("monitor", "cron(10/10 9-14 ? * MON-FRI *)", "장중 10분 감시"),
    ("monitor-late", "cron(0,10 15 ? * MON-FRI *)", "15:00·15:10 감시"),
    ("close", "cron(15 15 ? * MON-FRI *)", "15:15 마감 정리"),
]
SCHEDULE_TARGET = {"prep": "prep", "premarket": "premarket", "entry": "entry",
                   "monitor": "monitor", "monitor-late": "monitor", "close": "close"}

INCLUDE_DIRS = ["core", "jobs", "handlers"]
INCLUDE_FILES = ["config.py"]


def _session():
    return boto3.Session(region_name=config.AWS_REGION)


def _account_id() -> str:
    return _session().client("sts").get_caller_identity()["Account"]


def _fn_name(suffix: str) -> str:
    return f"{PROJECT}-{suffix}"


# ══════════════════════════════════════════════════════════════
def bootstrap() -> None:
    s = _session()
    ddb = s.client("dynamodb")
    iam = s.client("iam")

    # 상태 테이블 (아이템 1개짜리라 온디맨드로 충분하다)
    try:
        ddb.create_table(
            TableName=config.DYNAMODB_TABLE,
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"DynamoDB 테이블 생성: {config.DYNAMODB_TABLE}")
        ddb.get_waiter("table_exists").wait(TableName=config.DYNAMODB_TABLE)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"DynamoDB 테이블 이미 존재: {config.DYNAMODB_TABLE}")
        else:
            raise

    # 람다 실행 역할
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                       "Action": "sts:AssumeRole"}],
    }
    try:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust))
        print(f"IAM 역할 생성: {ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        print(f"IAM 역할 이미 존재: {ROLE_NAME}")

    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )

    acct = _account_id()
    inline = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"],
                "Resource": f"arn:aws:dynamodb:{config.AWS_REGION}:{acct}:table/{config.DYNAMODB_TABLE}",
            },
            {
                "Effect": "Allow",
                "Action": ["ssm:GetParameter", "ssm:PutParameter"],
                "Resource": f"arn:aws:ssm:{config.AWS_REGION}:{acct}:parameter{config.SSM_TOKEN_PATH}",
            },
        ],
    }
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName=f"{PROJECT}-state", PolicyDocument=json.dumps(inline))
    print("IAM 인라인 정책 적용 (DynamoDB + SSM)")

    # 스케줄러가 람다를 부르기 위한 역할
    sched_trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "scheduler.amazonaws.com"},
                       "Action": "sts:AssumeRole"}],
    }
    try:
        iam.create_role(RoleName=SCHED_ROLE_NAME, AssumeRolePolicyDocument=json.dumps(sched_trust))
        print(f"IAM 역할 생성: {SCHED_ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise

    iam.put_role_policy(
        RoleName=SCHED_ROLE_NAME,
        PolicyName=f"{PROJECT}-invoke",
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow", "Action": "lambda:InvokeFunction",
                "Resource": f"arn:aws:lambda:{config.AWS_REGION}:{acct}:function:{PROJECT}-*",
            }],
        }),
    )
    print("IAM 전파 대기 10초...")
    time.sleep(10)


# ══════════════════════════════════════════════════════════════
def build_zip() -> bytes:
    """소스 + requests 를 담은 배포 패키지."""
    import subprocess
    import tempfile

    vendor = tempfile.mkdtemp(prefix="ma5-vendor-")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "-r", os.path.join(ROOT, "requirements-lambda.txt"),
        "-t", vendor,
    ])

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, d)):
                dirnames[:] = [x for x in dirnames if x != "__pycache__"]
                for fn in filenames:
                    if not fn.endswith(".py"):
                        continue
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, ROOT))
        for f in INCLUDE_FILES:
            z.write(os.path.join(ROOT, f), f)

        for dirpath, dirnames, filenames in os.walk(vendor):
            dirnames[:] = [x for x in dirnames if x not in ("__pycache__", "bin")]
            for fn in filenames:
                if fn.endswith((".pyc", ".exe")):
                    continue
                full = os.path.join(dirpath, fn)
                z.write(full, os.path.relpath(full, vendor))

    data = buf.getvalue()
    print(f"배포 패키지 {len(data) / 1024 / 1024:.2f} MB")
    return data


def package() -> None:
    data = build_zip()
    out = os.path.join(ROOT, "build", "deploy.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)
    print("저장:", out)


def _env_vars() -> dict:
    return {
        "KIS_APP_KEY": config.KIS_APP_KEY,
        "KIS_APP_SECRET": config.KIS_APP_SECRET,
        "KIS_ACCOUNT_NO": config.KIS_ACCOUNT_NO,
        "EXCG_ID_DVSN_CD": config.EXCG_ID_DVSN_CD,
        "TELEGRAM_BOT_TOKEN": config.TELEGRAM_BOT_TOKEN,
        "TELEGRAM_ALLOWED_CHAT_IDS": ",".join(str(x) for x in config.TELEGRAM_ALLOWED_CHAT_IDS),
        "DRY_RUN": "true" if config.DRY_RUN else "false",
        # 람다는 상태를 공유해야 하므로 로컬 파일을 쓰면 안 된다
        "STATE_BACKEND": "dynamodb",
        "TOKEN_CACHE": "ssm",
        "DYNAMODB_TABLE": config.DYNAMODB_TABLE,
        # AWS_REGION 은 람다 런타임이 자동으로 넣어주는 예약 변수라 여기서 설정하면 거부된다.
    }


def lambdas() -> None:
    s = _session()
    lam = s.client("lambda")
    acct = _account_id()
    role_arn = f"arn:aws:iam::{acct}:role/{ROLE_NAME}"
    code = build_zip()

    for suffix, handler, timeout, memory in FUNCTIONS:
        name = _fn_name(suffix)
        try:
            lam.get_function(FunctionName=name)
            lam.update_function_code(FunctionName=name, ZipFile=code)
            waiter = lam.get_waiter("function_updated_v2")
            waiter.wait(FunctionName=name)
            lam.update_function_configuration(
                FunctionName=name, Handler=handler, Timeout=timeout,
                MemorySize=memory, Environment={"Variables": _env_vars()},
            )
            waiter.wait(FunctionName=name)
            print(f"갱신: {name}")
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            lam.create_function(
                FunctionName=name, Runtime=RUNTIME, Role=role_arn, Handler=handler,
                Code={"ZipFile": code}, Timeout=timeout, MemorySize=memory,
                Environment={"Variables": _env_vars()},
            )
            print(f"생성: {name}")


def schedules() -> None:
    s = _session()
    sch = s.client("scheduler")
    acct = _account_id()
    role_arn = f"arn:aws:iam::{acct}:role/{SCHED_ROLE_NAME}"

    for sid, expr, desc in SCHEDULES:
        name = f"{PROJECT}-{sid}"
        target_fn = _fn_name(SCHEDULE_TARGET[sid])
        params = dict(
            Name=name,
            ScheduleExpression=expr,
            ScheduleExpressionTimezone="Asia/Seoul",
            FlexibleTimeWindow={"Mode": "OFF"},
            Description=desc,
            Target={
                "Arn": f"arn:aws:lambda:{config.AWS_REGION}:{acct}:function:{target_fn}",
                "RoleArn": role_arn,
                "Input": json.dumps({"source": "scheduler"}),
                "RetryPolicy": {"MaximumRetryAttempts": 1},
            },
        )
        try:
            sch.create_schedule(**params)
            print(f"스케줄 생성: {name}  {expr}  ({desc})")
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise
            sch.update_schedule(**params)
            print(f"스케줄 갱신: {name}  {expr}  ({desc})")


def webhook() -> None:
    s = _session()
    api = s.client("apigatewayv2")
    lam = s.client("lambda")
    acct = _account_id()
    fn = _fn_name("webhook")
    fn_arn = f"arn:aws:lambda:{config.AWS_REGION}:{acct}:function:{fn}"

    existing = next(
        (x for x in api.get_apis()["Items"] if x["Name"] == f"{PROJECT}-webhook"), None
    )
    if existing:
        api_id = existing["ApiId"]
        endpoint = existing["ApiEndpoint"]
        print(f"API 이미 존재: {api_id}")
    else:
        created = api.create_api(
            Name=f"{PROJECT}-webhook", ProtocolType="HTTP", Target=fn_arn,
        )
        api_id, endpoint = created["ApiId"], created["ApiEndpoint"]
        print(f"API 생성: {api_id}")

    try:
        lam.add_permission(
            FunctionName=fn, StatementId="apigw-invoke",
            Action="lambda:InvokeFunction", Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{config.AWS_REGION}:{acct}:{api_id}/*/*",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceConflictException":
            raise

    if not config.TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN 이 없어 웹훅 등록은 생략한다.")
        print("엔드포인트:", endpoint)
        return

    import requests
    r = requests.post(
        f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/setWebhook",
        json={"url": endpoint, "allowed_updates": ["message"]}, timeout=10,
    )
    print("텔레그램 웹훅 등록:", r.json())
    print("엔드포인트:", endpoint)


COMMANDS = {
    "bootstrap": bootstrap,
    "package": package,
    "lambdas": lambdas,
    "schedules": schedules,
    "webhook": webhook,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "all"):
        print(__doc__)
        return 1

    problems = config.validate()
    if problems:
        print("설정 문제로 중단한다:")
        for p in problems:
            print("  -", p)
        return 1

    cmd = sys.argv[1]
    if cmd == "all":
        for name in ("bootstrap", "lambdas", "schedules", "webhook"):
            print(f"\n─── {name} ───")
            COMMANDS[name]()
    else:
        COMMANDS[cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
