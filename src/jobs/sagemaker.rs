//! AWS SageMaker Training Jobs backend — one training job per run, on the user's AWS account.
//!
//! Driven through `boto3` in an orx-managed venv, like Modal and Vercel: every operation
//! shells out to a bundled launcher. boto3 resolves the standard credential chain (keys,
//! `AWS_PROFILE`, IAM Identity Center, instance roles), so orx stores no AWS secret of its own.
//!
//! A training job releases its instance when it completes, fails, hits
//! `MaxRuntimeInSeconds` or is stopped, so nothing keeps billing if orx goes away. The
//! snapshot, the run script and the run's env file travel through one S3 input channel:
//! the job's `Environment` is readable by anyone who can describe it, and an entrypoint
//! item is capped at 256 characters. `finish` deletes the env file once the run is over.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{json, Value};

use super::managed_env::ManagedEnv;
use crate::error::{anyhow, Result};

/// Bundled Python launcher. Subcommands (argv[1]):
///   check                      config JSON on stdin -> {"account", "arn", "region"}
///   submit                     spec JSON on stdin -> {"jobName", "region", "inputUri"}
///   status <name> <region>     -> {"stage", "region", "message"?, "exitCode"?, "startedAt"?, ...}
///   logs   <name> <region>     replays the job's CloudWatch log from the start until it ends
///   cancel <name> <region>     stop the training job
///   finish <region> <input>    delete the run's env file from S3
///
/// Exit 3 = `boto3` isn't importable, 6 = refused (`jobs::Refused` on stderr),
/// 7 = no AWS credentials found.
const LAUNCHER: &str = r#"
import sys, json, os, re, time, shlex

GROUP = "/aws/sagemaker/TrainingJobs"
CHANNEL = "orx"
INPUT = "/opt/ml/input/data/" + CHANNEL
TERMINAL = ("Completed", "Failed", "Stopped")
QUEUED = ("Starting", "Pending", "LaunchingMLInstances", "PreparingTrainingStack",
          "Downloading", "DownloadingTrainingImage")
DLC_ACCOUNT = "763104351884"
DLC_REGIONS = ("us-east-1", "us-east-2", "us-west-1", "us-west-2", "ca-central-1", "sa-east-1",
               "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-north-1",
               "ap-northeast-1", "ap-northeast-2", "ap-northeast-3", "ap-southeast-1",
               "ap-southeast-2", "ap-south-1")
GPU_FAMILIES = ("g4dn", "g5", "g6", "g6e", "p3", "p3dn", "p4d", "p4de", "p5", "p5e", "p5en")
GPU_IMAGE = "pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker"
CPU_IMAGE = "pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker"

def _sdk():
    try:
        import boto3, botocore.exceptions as errors
        return boto3, errors
    except Exception:
        sys.stderr.write("boto3-not-installed")
        sys.exit(3)

def _refuse(status, message):
    sys.stderr.write(json.dumps({"status": status, "message": message}))
    sys.exit(6)

def _session(region):
    boto3, errors = _sdk()
    # boto3 reads AWS_DEFAULT_REGION but not AWS_REGION, which the other AWS SDKs use.
    session = boto3.session.Session(region_name=region or os.environ.get("AWS_REGION") or None)
    if not session.region_name:
        _refuse(None, "No AWS region is set. Set ORX_SAGEMAKER_REGION (or AWS_REGION), or a "
                      "region in your AWS profile.")
    if session.get_credentials() is None:
        sys.stderr.write("no-credentials")
        sys.exit(7)
    return session, errors

def _code(error):
    return (getattr(error, "response", None) or {}).get("Error", {}).get("Code", "")

def _http(error):
    return (getattr(error, "response", None) or {}).get("ResponseMetadata", {}).get("HTTPStatusCode")

def _missing(error):
    return _code(error) == "ValidationException" and "not found" in str(error).lower()

def _split_s3(uri):
    m = re.match(r"^s3://([^/]+)/?(.*)$", uri or "")
    if not m:
        _refuse(None, "%r is not an s3://bucket/prefix URI." % uri)
    return m.group(1), m.group(2).strip("/")

def _ms(value):
    return int(value.timestamp() * 1000) if value is not None else None

def _default_image(region, instance_type):
    if region not in DLC_REGIONS:
        _refuse(None, "orx has no default image for region %s. Pass --image with a training "
                      "image this region can pull." % region)
    family = instance_type.split(".")[1] if instance_type.count(".") >= 2 else ""
    name = GPU_IMAGE if family in GPU_FAMILIES else CPU_IMAGE
    return "%s.dkr.ecr.%s.amazonaws.com/%s" % (DLC_ACCOUNT, region, name)

def check():
    spec = json.load(sys.stdin)
    session, errors = _session(spec.get("region"))
    identity = session.client("sts").get_caller_identity()
    bucket, _ = _split_s3(spec["s3Uri"])
    try:
        session.client("s3").head_bucket(Bucket=bucket)
    except errors.ClientError as error:
        _refuse(_http(error), "Can't reach S3 bucket %s with these credentials (%s)."
                % (bucket, _code(error) or error))
    print(json.dumps({"account": identity["Account"], "arn": identity["Arn"],
                      "region": session.region_name}))

def submit():
    spec = json.load(sys.stdin)
    session, errors = _session(spec.get("region"))
    region = session.region_name
    image = spec.get("image") or _default_image(region, spec["instanceType"])
    bucket, prefix = _split_s3(spec["s3Uri"])
    base = "/".join(p for p in (prefix, "orx", spec["name"]) if p)
    s3 = session.client("s3")
    env = spec.get("env") or {}
    body = "".join("export %s=%s\n" % (k, shlex.quote(str(v))) for k, v in sorted(env.items()))
    run = "set -a; . %s/env.sh; set +a; mkdir -p /opt/orx; cd /opt/orx\n%s" % (INPUT, spec["script"])
    try:
        if spec.get("sourceArchive"):
            s3.upload_file(spec["sourceArchive"], bucket, base + "/input/source.tar")
            s3.put_object(Bucket=bucket, Key=base + "/input/source.tar.ready", Body=b"")
        s3.put_object(Bucket=bucket, Key=base + "/input/env.sh", Body=body.encode("utf-8"),
                      ServerSideEncryption="AES256")
        s3.put_object(Bucket=bucket, Key=base + "/input/run.sh", Body=run.encode("utf-8"))
    except errors.ClientError as error:
        _refuse(_http(error), "Could not stage the run in s3://%s/%s: %s" % (bucket, base, error))
    input_uri = "s3://%s/%s/input/" % (bucket, base)
    try:
        session.client("sagemaker").create_training_job(
            TrainingJobName=spec["name"],
            RoleArn=spec["roleArn"],
            AlgorithmSpecification={
                "TrainingImage": image,
                "TrainingInputMode": "File",
                "ContainerEntrypoint": ["bash", INPUT + "/run.sh"],
            },
            InputDataConfig=[{
                "ChannelName": CHANNEL,
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }},
            }],
            OutputDataConfig={"S3OutputPath": "s3://%s/%s/output" % (bucket, base)},
            ResourceConfig={
                "InstanceType": spec["instanceType"],
                "InstanceCount": 1,
                "VolumeSizeInGB": int(spec["volumeGb"]),
            },
            StoppingCondition={"MaxRuntimeInSeconds": int(spec["timeoutSeconds"])},
            Tags=[{"Key": k, "Value": str(v)} for k, v in sorted((spec.get("tags") or {}).items())],
        )
    except errors.ClientError as error:
        _refuse(_http(error), "SageMaker didn't create training job %s: %s" % (spec["name"], error))
    print(json.dumps({"jobName": spec["name"], "region": region, "inputUri": input_uri}))

def _describe(sm, errors, name):
    try:
        return sm.describe_training_job(TrainingJobName=name)
    except errors.ClientError as error:
        if _missing(error):
            return None
        raise

def status(name, region):
    session, errors = _session(region)
    job = _describe(session.client("sagemaker"), errors, name)
    if job is None:
        print(json.dumps({"missing": True}))
        return
    state, secondary = job["TrainingJobStatus"], job.get("SecondaryStatus", "")
    out = {"region": session.region_name, "startedAt": _ms(job.get("TrainingStartTime")),
           "finishedAt": _ms(job.get("TrainingEndTime"))}
    if state == "Completed":
        out.update(stage="COMPLETED", exitCode=0)
    elif state == "Failed":
        reason = job.get("FailureReason") or "the training job failed"
        out.update(stage="ERROR", message=reason)
        code = re.search(r"exit code:?\s*(\d+)", reason, re.I)
        if code:
            out["exitCode"] = int(code.group(1))
    elif state == "Stopped" and secondary == "MaxRuntimeExceeded":
        limit = job.get("StoppingCondition", {}).get("MaxRuntimeInSeconds")
        out.update(stage="ERROR", message="timed out: SageMaker stopped the job at its %ss limit" % limit)
    elif state == "Stopped":
        out.update(stage="CANCELED")
    elif state == "InProgress" and secondary in QUEUED:
        out.update(stage="SCHEDULING", message=job.get("SecondaryStatusTransitions", [{}])[-1].get("StatusMessage") or secondary)
    else:
        out.update(stage="RUNNING")
    print(json.dumps(out))

def logs(name, region):
    session, errors = _session(region)
    sm, cw = session.client("sagemaker"), session.client("logs")
    tokens, quiet_after_end = {}, 0
    while True:
        try:
            streams = cw.describe_log_streams(logGroupName=GROUP, logStreamNamePrefix=name + "/")["logStreams"]
        except errors.ClientError as error:
            if _code(error) != "ResourceNotFoundException":
                raise
            streams = []
        got = False
        for stream in sorted(s["logStreamName"] for s in streams):
            kwargs = {"logGroupName": GROUP, "logStreamName": stream, "startFromHead": True}
            if stream in tokens:
                kwargs["nextToken"] = tokens[stream]
            while True:
                page = cw.get_log_events(**kwargs)
                for event in page["events"]:
                    got = True
                    sys.stdout.write(event["message"].rstrip("\n") + "\n")
                sys.stdout.flush()
                token = page.get("nextForwardToken")
                if not token or token == kwargs.get("nextToken"):
                    break
                kwargs["nextToken"] = token
            if "nextToken" in kwargs:
                tokens[stream] = kwargs["nextToken"]
        if not got:
            job = _describe(sm, errors, name)
            if job is None or job["TrainingJobStatus"] in TERMINAL:
                quiet_after_end += 1
                if quiet_after_end > 1:
                    return
        time.sleep(2)

def cancel(name, region):
    session, errors = _session(region)
    sm = session.client("sagemaker")
    job = _describe(sm, errors, name)
    if job is None:
        print(json.dumps({"missing": True}))
        return
    if job["TrainingJobStatus"] == "InProgress":
        sm.stop_training_job(TrainingJobName=name)
    print(json.dumps({"missing": False}))

def finish(region, input_uri):
    session, _ = _session(region)
    bucket, prefix = _split_s3(input_uri)
    session.client("s3").delete_object(Bucket=bucket, Key=prefix + "/env.sh")

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "check":
    check()
elif cmd == "submit":
    submit()
elif cmd == "status":
    status(sys.argv[2], sys.argv[3])
elif cmd == "logs":
    logs(sys.argv[2], sys.argv[3])
elif cmd == "cancel":
    cancel(sys.argv[2], sys.argv[3])
elif cmd == "finish":
    finish(sys.argv[2], sys.argv[3])
else:
    sys.stderr.write("unknown launcher subcommand: %r" % cmd)
    sys.exit(2)
"#;

pub(crate) const SAGEMAKER_ENV: ManagedEnv = ManagedEnv {
    name: "sagemaker",
    label: "AWS SageMaker",
    min_python: (3, 10),
    requirement: "boto3>=1.34",
    ready_check: "import boto3, botocore.exceptions",
    override_var: Some("ORX_SAGEMAKER_PYTHON"),
};

pub const ROLE_ARN: &str = "ORX_SAGEMAKER_ROLE_ARN";
pub const S3_URI: &str = "ORX_SAGEMAKER_S3_URI";
pub const REGION: &str = "ORX_SAGEMAKER_REGION";
/// What boto3 reads from its environment; a saved value reaches the launcher, never the job.
const AWS_KEYS: [&str; 6] = [
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
];

pub const DEFAULT_TIMEOUT_SECS: u64 = 4 * 3600;
/// SageMaker's own ceiling for `MaxRuntimeInSeconds`.
pub const MAX_TIMEOUT_SECS: u64 = 28 * 24 * 3600;
pub const VOLUME_GB: u32 = 50;

pub async fn ensure_env() -> Result<()> {
    if SAGEMAKER_ENV.override_python().is_some() {
        return Ok(());
    }
    SAGEMAKER_ENV.ensure().await.map_err(|e| {
        anyhow!(
            "{}. Or set ORX_SAGEMAKER_PYTHON to an interpreter with {}.",
            e.to_string().trim_end_matches('.'),
            SAGEMAKER_ENV.requirement
        )
    })?;
    Ok(())
}

/// A non-empty process env value wins, then the synced env file (Settings → Environment).
fn resolve(
    key: &str,
    process: impl Fn(&str) -> Option<String>,
    synced: impl Fn(&str) -> Option<String>,
) -> Option<(String, &'static str)> {
    let present = |v: &String| !v.trim().is_empty();
    process(key)
        .filter(present)
        .map(|v| (v.trim().to_string(), "env"))
        .or_else(|| {
            synced(key)
                .filter(present)
                .map(|v| (v.trim().to_string(), "syncedEnv"))
        })
}

pub fn setting(key: &str) -> Option<(String, &'static str)> {
    resolve(
        key,
        |k| std::env::var(k).ok(),
        crate::config::synced_env_var,
    )
}

fn launcher_overrides(
    process: impl Fn(&str) -> Option<String>,
    synced: impl Fn(&str) -> Option<String>,
) -> Vec<(String, String)> {
    AWS_KEYS
        .into_iter()
        .filter_map(|key| match resolve(key, &process, &synced) {
            Some((value, "syncedEnv")) => Some((key.to_string(), value)),
            _ => None,
        })
        .collect()
}

fn launcher_env() -> Vec<(String, String)> {
    launcher_overrides(|k| std::env::var(k).ok(), crate::config::synced_env_var)
}

/// An empty inherited value is unset to orx but names a profile "" to boto3.
fn blank_aws_keys() -> Vec<&'static str> {
    AWS_KEYS
        .into_iter()
        .filter(|key| std::env::var(key).is_ok_and(|v| v.trim().is_empty()))
        .collect()
}

/// The job runs as the execution role; it gets neither the caller's AWS keys nor orx's settings.
pub fn strip_credentials(env: &mut HashMap<String, String>) {
    for key in AWS_KEYS.into_iter().chain([ROLE_ARN, S3_URI, REGION]) {
        env.remove(key);
    }
}

pub fn check_role_arn(value: &str) -> Result<()> {
    let ok = value
        .strip_prefix("arn:")
        .and_then(|rest| rest.split_once(":iam::"))
        .and_then(|(partition, rest)| {
            partition
                .starts_with("aws")
                .then_some(rest)?
                .split_once(":role/")
        })
        .is_some_and(|(account, name)| {
            account.len() == 12 && account.bytes().all(|b| b.is_ascii_digit()) && !name.is_empty()
        });
    if ok {
        Ok(())
    } else {
        Err(anyhow!(
            "{ROLE_ARN} must be an IAM role ARN like arn:aws:iam::123456789012:role/orx-sagemaker \
             (got '{value}')."
        ))
    }
}

pub fn check_s3_uri(value: &str) -> Result<()> {
    let bucket = value
        .strip_prefix("s3://")
        .map(|rest| rest.split('/').next().unwrap_or(""));
    match bucket {
        Some(bucket) if (3..=63).contains(&bucket.len()) => Ok(()),
        _ => Err(anyhow!(
            "{S3_URI} must be an S3 location like s3://my-bucket/orx (got '{value}')."
        )),
    }
}

/// The settings a launch needs, checked for shape only; `check` proves they work.
pub struct Config {
    pub role_arn: String,
    pub s3_uri: String,
    pub region: Option<String>,
}

pub fn config() -> Result<Config> {
    let missing = |key: &str| {
        anyhow!(
            "AWS SageMaker isn't set up: {key} is missing. Set it in Settings → Compute → AWS \
             SageMaker or in the environment."
        )
    };
    let role_arn = setting(ROLE_ARN).ok_or_else(|| missing(ROLE_ARN))?.0;
    let s3_uri = setting(S3_URI).ok_or_else(|| missing(S3_URI))?.0;
    check_role_arn(&role_arn)?;
    check_s3_uri(&s3_uri)?;
    Ok(Config {
        role_arn,
        s3_uri,
        region: setting(REGION).map(|(v, _)| v),
    })
}

fn missing_python_hint(e: &std::io::Error) -> crate::error::Error {
    if e.kind() == std::io::ErrorKind::NotFound {
        anyhow!(
            "{} not found on PATH — AWS SageMaker is driven through boto3. Install Python 3.10+, \
             or point orx at an interpreter with boto3 via ORX_SAGEMAKER_PYTHON.",
            SAGEMAKER_ENV.interpreter()
        )
    } else {
        anyhow!("Could not run {}: {}", SAGEMAKER_ENV.interpreter(), e)
    }
}

fn launcher_error(what: &str, code: Option<i32>, stderr: &str) -> crate::error::Error {
    let stderr = stderr.trim();
    if code == Some(3) || stderr == "boto3-not-installed" {
        return anyhow!(
            "The `boto3` Python package isn't importable by {}. Remove {} to force a rebuild, or \
             set ORX_SAGEMAKER_PYTHON to an interpreter that has it.",
            SAGEMAKER_ENV.interpreter(),
            SAGEMAKER_ENV.dir().display()
        );
    }
    if code == Some(7) || stderr == "no-credentials" {
        return anyhow!(
            "boto3 found no AWS credentials. Sign in (`aws sso login --profile <name>`) and set \
             AWS_PROFILE, or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, in the environment \
             or Settings → Environment."
        );
    }
    let last = stderr.lines().last().unwrap_or("");
    if let Some(refused) = super::launcher_refusal(code, last) {
        return refused.into();
    }
    let detail = if last.is_empty() { "no stderr" } else { last };
    anyhow!(
        "sagemaker {} failed{}: {}",
        what,
        code.map(|c| format!(" (exit {c})")).unwrap_or_default(),
        detail
    )
}

async fn launcher_capture(
    args: &[&str],
    stdin: Option<&str>,
    fence: Option<&crate::engine::Fence<'_>>,
) -> Result<Vec<u8>> {
    let mut cmd = crate::process::tokio_command(SAGEMAKER_ENV.interpreter());
    for key in blank_aws_keys() {
        cmd.env_remove(key);
    }
    cmd.arg("-c")
        .arg(LAUNCHER)
        .args(args)
        .env(super::PYTHONUNBUFFERED, "1")
        .envs(launcher_env())
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| missing_python_hint(&e))?;
    if let Some(fence) = fence {
        fence.record_child(&mut child).await?;
    }
    if let Some(input) = stdin {
        use tokio::io::AsyncWriteExt as _;
        if let Some(mut pipe) = child.stdin.take() {
            pipe.write_all(input.as_bytes())
                .await
                .map_err(|e| anyhow!("Could not send the sagemaker launcher its input: {e}"))?;
        }
    }
    let out = child
        .wait_with_output()
        .await
        .map_err(|e| anyhow!("Could not run {}: {}", SAGEMAKER_ENV.interpreter(), e))?;
    if !out.status.success() {
        return Err(launcher_error(
            args.first().copied().unwrap_or("?"),
            out.status.code(),
            &String::from_utf8_lossy(&out.stderr),
        ));
    }
    Ok(out.stdout)
}

fn parse<T: serde::de::DeserializeOwned>(what: &str, out: &[u8]) -> Result<T> {
    serde_json::from_slice(out).map_err(|e| {
        anyhow!(
            "Unreadable sagemaker {what} output: {e} ({:?})",
            String::from_utf8_lossy(out)
        )
    })
}

/// `--flavor` is a SageMaker instance type, with or without the `ml.` prefix.
pub fn resolve_flavor(name: Option<&str>) -> Result<String> {
    let raw = name
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "AWS SageMaker requires --flavor: an instance type such as ml.g5.xlarge (A10G), \
             ml.g6.xlarge (L4), ml.g4dn.xlarge (T4) or ml.m5.large (CPU)."
            )
        })?;
    let lower = raw.to_ascii_lowercase();
    let bare = lower.strip_prefix("ml.").unwrap_or(&lower);
    let valid = bare.split_once('.').is_some_and(|(family, size)| {
        let word = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric());
        word(family) && word(size) && size.ends_with("large")
    });
    if valid {
        Ok(format!("ml.{bare}"))
    } else {
        Err(anyhow!(
            "Unknown AWS SageMaker flavor '{raw}'. Use an instance type such as ml.g5.xlarge, \
             ml.g6.2xlarge or ml.m5.large."
        ))
    }
}

pub fn check_timeout(seconds: u64) -> Result<()> {
    if seconds == 0 || seconds > MAX_TIMEOUT_SECS {
        return Err(anyhow!(
            "AWS SageMaker runs a training job for at most 28 days; pass a --timeout between 1s \
             and 28d."
        ));
    }
    Ok(())
}

/// A training job name for a run: unique per account and region, never reusable.
pub fn job_name(run_id: &str) -> String {
    let mut name = String::from("orx-");
    for c in run_id.chars() {
        name.push(if c.is_ascii_alphanumeric() { c } else { '-' });
    }
    name.truncate(63);
    name.trim_end_matches('-').to_string()
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SageMakerJobSpec {
    pub name: String,
    pub script: String,
    pub image: Option<String>,
    pub instance_type: String,
    pub env: HashMap<String, String>,
    pub timeout_seconds: u64,
    pub volume_gb: u32,
    pub role_arn: String,
    pub s3_uri: String,
    pub region: Option<String>,
    pub tags: HashMap<String, String>,
    pub source_archive: Option<PathBuf>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Submitted {
    pub job_name: String,
    pub region: String,
    pub input_uri: String,
}

/// Stage the input channel and create the training job. The launcher waits for its spec on
/// stdin, so it creates nothing before it is recorded on `fence`.
pub async fn run_job(
    spec: &SageMakerJobSpec,
    fence: Option<&crate::engine::Fence<'_>>,
) -> Result<Submitted> {
    let body = json!({
        "name": spec.name,
        "script": spec.script,
        "image": spec.image,
        "instanceType": spec.instance_type,
        "env": super::default_python_env(&spec.env),
        "timeoutSeconds": spec.timeout_seconds,
        "volumeGb": spec.volume_gb,
        "roleArn": spec.role_arn,
        "s3Uri": spec.s3_uri,
        "region": spec.region,
        "tags": spec.tags,
        "sourceArchive": spec.source_archive,
    });
    let out = launcher_capture(&["submit"], Some(&body.to_string()), fence).await?;
    parse("submit", &out)
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobState {
    #[serde(default)]
    pub missing: bool,
    #[serde(default)]
    pub stage: String,
    pub message: Option<String>,
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub region: String,
    pub started_at: Option<i64>,
    pub finished_at: Option<i64>,
}

/// `None` when SageMaker has no job by that name in that region. An empty region is the
/// profile's own, as at submit.
pub async fn inspect_job(name: &str, region: &str) -> Result<Option<JobState>> {
    let out = launcher_capture(&["status", name, region], None, None).await?;
    let state: JobState = parse("status", &out)?;
    Ok((!state.missing).then_some(state))
}

/// False when SageMaker has no job by that name in that region.
pub async fn cancel_job(name: &str, region: &str) -> Result<bool> {
    let out = launcher_capture(&["cancel", name, region], None, None).await?;
    let v: Value = parse("cancel", &out)?;
    Ok(v["missing"] != true)
}

pub async fn finish_job(region: &str, input_uri: &str) -> Result<()> {
    launcher_capture(&["finish", region, input_uri], None, None).await?;
    Ok(())
}

/// Who the credentials are, proven against STS and the staging bucket.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Identity {
    pub account: String,
    pub arn: String,
    pub region: String,
}

pub async fn check(config: &Config) -> Result<Identity> {
    let body = json!({"s3Uri": config.s3_uri, "region": config.region});
    let out = launcher_capture(&["check"], Some(&body.to_string()), None).await?;
    parse("check", &out)
}

/// One pass over the job's log, invoking `sink` per line past `skip`. Each connect replays
/// the log from its first line; ends when the job is over or after `idle` silence.
pub async fn stream_logs(
    name: &str,
    region: &str,
    skip: u64,
    idle: Duration,
    sink: &mut (dyn FnMut(&str) + Send),
) -> Result<u64> {
    use tokio::io::{AsyncBufReadExt as _, AsyncReadExt as _, BufReader};

    let mut cmd = crate::process::tokio_command(SAGEMAKER_ENV.interpreter());
    for key in blank_aws_keys() {
        cmd.env_remove(key);
    }
    let mut child = cmd
        .arg("-c")
        .arg(LAUNCHER)
        .args(["logs", name, region])
        .env(super::PYTHONUNBUFFERED, "1")
        .envs(launcher_env())
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| missing_python_hint(&e))?;
    let stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take();
    let mut lines = BufReader::new(stdout).lines();
    let mut seen = 0u64;
    loop {
        match tokio::time::timeout(idle, lines.next_line()).await {
            Err(_) => break,
            Ok(Err(e)) => return Err(anyhow!("Could not read the SageMaker log launcher: {e}")),
            Ok(Ok(None)) => {
                let status = child.wait().await?;
                if !status.success() {
                    let mut detail = String::new();
                    if let Some(stderr) = stderr.as_mut() {
                        stderr.read_to_string(&mut detail).await?;
                    }
                    return Err(launcher_error("logs", status.code(), &detail));
                }
                break;
            }
            Ok(Ok(Some(line))) => {
                seen += 1;
                if seen > skip {
                    sink(&line);
                }
            }
        }
    }
    let _ = child.kill().await;
    Ok(seen.max(skip))
}

/// Fail fast before staging if SageMaker isn't usable: settings, boto3, credentials, bucket.
pub async fn preflight(
    caller_overrides: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    SAGEMAKER_ENV.require_caller_override(caller_overrides)?;
    let config = config()?;
    ensure_env().await?;
    check(&config).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn flavors_are_instance_types() {
        assert_eq!(
            resolve_flavor(Some("ml.g5.xlarge")).unwrap(),
            "ml.g5.xlarge"
        );
        assert_eq!(resolve_flavor(Some("G6.2XLARGE")).unwrap(), "ml.g6.2xlarge");
        assert_eq!(resolve_flavor(Some(" m5.large ")).unwrap(), "ml.m5.large");
        for bad in ["", "a10g", "ml.g5", "g5.huge", "ml.g5.x large", "h100:2"] {
            assert!(resolve_flavor(Some(bad)).is_err(), "{bad}");
        }
        assert!(resolve_flavor(None)
            .unwrap_err()
            .to_string()
            .contains("--flavor"));
    }

    #[test]
    fn job_names_are_stable_and_valid() {
        assert_eq!(job_name("r_01ABC"), "orx-r-01ABC");
        let long = job_name(&"x".repeat(100));
        assert_eq!(long.len(), 63);
        assert!(!job_name("run_").ends_with('-'));
    }

    #[test]
    fn timeouts_stay_within_the_sagemaker_ceiling() {
        assert!(check_timeout(DEFAULT_TIMEOUT_SECS).is_ok());
        assert!(check_timeout(MAX_TIMEOUT_SECS).is_ok());
        assert!(check_timeout(MAX_TIMEOUT_SECS + 1).is_err());
        assert!(check_timeout(0).is_err());
    }

    #[test]
    fn settings_are_checked_for_shape() {
        assert!(check_role_arn("arn:aws:iam::123456789012:role/orx").is_ok());
        assert!(check_role_arn("arn:aws-cn:iam::123456789012:role/path/orx").is_ok());
        for bad in [
            "orx",
            "arn:aws:iam::1234:role/orx",
            "arn:aws:iam::123456789012:user/orx",
            "arn:aws:iam::123456789012:role/",
        ] {
            assert!(check_role_arn(bad).is_err(), "{bad}");
        }
        assert!(check_s3_uri("s3://my-bucket").is_ok());
        assert!(check_s3_uri("s3://my-bucket/orx/runs").is_ok());
        for bad in [
            "my-bucket",
            "s3://",
            "s3://ab/x",
            "https://my-bucket.s3.amazonaws.com",
        ] {
            assert!(check_s3_uri(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn launcher_gets_saved_aws_keys_only_where_the_process_lacks_them() {
        let process = |k: &str| match k {
            "AWS_PROFILE" => Some("research".to_string()),
            "AWS_REGION" => Some("  ".to_string()),
            _ => None,
        };
        let synced = |k: &str| match k {
            "AWS_PROFILE" => Some("other".to_string()),
            "AWS_REGION" => Some("sa-east-1".to_string()),
            "HF_TOKEN" => Some("hf".to_string()),
            _ => None,
        };
        assert_eq!(
            launcher_overrides(process, synced),
            vec![("AWS_REGION".to_string(), "sa-east-1".to_string())]
        );
    }

    #[test]
    fn job_env_drops_aws_credentials_and_orx_settings() {
        let mut env = HashMap::from([
            ("AWS_SECRET_ACCESS_KEY".to_string(), "x".to_string()),
            ("AWS_PROFILE".to_string(), "x".to_string()),
            (ROLE_ARN.to_string(), "x".to_string()),
            ("HF_TOKEN".to_string(), "kept".to_string()),
        ]);
        strip_credentials(&mut env);
        assert_eq!(env.keys().collect::<Vec<_>>(), vec!["HF_TOKEN"]);
    }

    #[test]
    fn launcher_failures_name_their_cause() {
        assert!(launcher_error("submit", Some(3), "")
            .to_string()
            .contains("boto3"));
        assert!(launcher_error("status", Some(7), "no-credentials")
            .to_string()
            .contains("aws sso login"));
        let refused = launcher_error(
            "submit",
            Some(super::super::LAUNCHER_REFUSED),
            r#"{"status": 400, "message": "SageMaker didn't create training job orx-r: quota"}"#,
        );
        assert!(refused.downcast_ref::<super::super::Refused>().is_some());
        let other = launcher_error(
            "logs",
            Some(1),
            "Traceback\nbotocore.exceptions.EndpointConnectionError: boom",
        );
        assert!(
            other.to_string().contains("EndpointConnectionError: boom"),
            "{other}"
        );
    }

    const FAKE_BOTO3: &str = r#"import sys, json, datetime
class _Creds: pass
def _emit(event): sys.stderr.write(json.dumps(event, default=str) + "\n")
class _Client:
    def __init__(self, service, state): self.service, self.state = service, state
    def get_caller_identity(self): return {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/x/y"}
    def head_bucket(self, Bucket): _emit({"head": Bucket})
    def upload_file(self, path, bucket, key): _emit({"upload": key})
    def put_object(self, Bucket, Key, Body, **kw): _emit({"put": Key, "body": Body.decode(), "sse": kw.get("ServerSideEncryption")})
    def delete_object(self, Bucket, Key): _emit({"delete": Key})
    def create_training_job(self, **kw): _emit({"create": kw})
    def stop_training_job(self, TrainingJobName): _emit({"stop": TrainingJobName})
    def describe_training_job(self, TrainingJobName):
        job = json.loads(self.state)
        if job is None:
            import botocore.exceptions as e
            raise e.ClientError({"Error": {"Code": "ValidationException", "Message": "Requested resource not found."}}, "DescribeTrainingJob")
        for key in ("TrainingStartTime", "TrainingEndTime"):
            if key in job: job[key] = datetime.datetime.fromtimestamp(job[key], datetime.timezone.utc)
        return job
    def describe_log_streams(self, **kw): return {"logStreams": [{"logStreamName": "orx-r/algo-1-1"}]}
    def get_log_events(self, **kw):
        if "nextToken" in kw: return {"events": [], "nextForwardToken": "t1"}
        return {"events": [{"message": "step 1"}, {"message": "step 2\n"}], "nextForwardToken": "t1"}
class _Session:
    def __init__(self, region_name=None):
        import os
        self.region_name = region_name or os.environ.get("FAKE_REGION")
    def get_credentials(self):
        import os
        return None if os.environ.get("FAKE_NO_CREDS") else _Creds()
    def client(self, service):
        import os
        return _Client(service, os.environ.get("FAKE_JOB", "null"))
class session:
    Session = _Session
"#;

    const FAKE_BOTOCORE: &str = r#"class ClientError(Exception):
    def __init__(self, response, op):
        super().__init__(response["Error"]["Message"])
        self.response = response
"#;

    struct Fake {
        dir: PathBuf,
        python: &'static str,
    }

    impl Fake {
        async fn new() -> Option<Self> {
            let python = crate::jobs::managed_env::base_python(
                SAGEMAKER_ENV.min_python,
                SAGEMAKER_ENV.label,
            )
            .await
            .ok()?;
            let dir = std::env::temp_dir().join(format!("orx-sagemaker-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(dir.join("boto3")).unwrap();
            std::fs::create_dir_all(dir.join("botocore")).unwrap();
            std::fs::write(dir.join("boto3").join("__init__.py"), FAKE_BOTO3).unwrap();
            std::fs::write(dir.join("botocore").join("__init__.py"), "").unwrap();
            std::fs::write(dir.join("botocore").join("exceptions.py"), FAKE_BOTOCORE).unwrap();
            Some(Self { dir, python })
        }

        fn run(&self, args: &[&str], stdin: &str, env: &[(&str, &str)]) -> std::process::Output {
            let mut child = crate::process::command(self.python)
                .args(["-c", LAUNCHER])
                .args(args)
                .env("PYTHONPATH", &self.dir)
                .envs(env.iter().copied())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(stdin.as_bytes())
                .unwrap();
            child.wait_with_output().unwrap()
        }
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn events(out: &std::process::Output) -> Vec<Value> {
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    #[tokio::test]
    async fn submit_keeps_secrets_out_of_the_training_job() {
        let Some(fake) = Fake::new().await else {
            crate::config::skip_without_tool("no Python 3.10+ on PATH");
            return;
        };
        let spec = json!({
            "name": "orx-r", "script": "python train.py", "image": null,
            "instanceType": "ml.g5.xlarge", "env": {"HF_TOKEN": "s3cr'et"},
            "timeoutSeconds": 3600, "volumeGb": 50,
            "roleArn": "arn:aws:iam::123456789012:role/orx", "s3Uri": "s3://bucket/research/",
            "region": "sa-east-1", "tags": {"or_run": "r"}, "sourceArchive": "/tmp/src.tar",
        });
        let out = fake.run(&["submit"], &spec.to_string(), &[]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&out.stdout).unwrap(),
            json!({"jobName": "orx-r", "region": "sa-east-1",
                   "inputUri": "s3://bucket/research/orx/orx-r/input/"})
        );
        let events = events(&out);
        assert_eq!(events[0]["upload"], "research/orx/orx-r/input/source.tar");
        let env_file = events
            .iter()
            .find(|e| e["put"] == "research/orx/orx-r/input/env.sh")
            .unwrap();
        assert_eq!(env_file["body"], "export HF_TOKEN='s3cr'\"'\"'et'\n");
        assert_eq!(env_file["sse"], "AES256");
        let run = events
            .iter()
            .find(|e| e["put"] == "research/orx/orx-r/input/run.sh")
            .unwrap();
        assert!(
            run["body"].as_str().unwrap().ends_with("\npython train.py"),
            "{run}"
        );
        let create = &events.iter().find(|e| e.get("create").is_some()).unwrap()["create"];
        assert!(!create.to_string().contains("s3cr"), "{create}");
        assert!(create.get("Environment").is_none(), "{create}");
        assert_eq!(
            create["AlgorithmSpecification"]["TrainingImage"],
            "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker"
        );
        assert_eq!(create["StoppingCondition"]["MaxRuntimeInSeconds"], 3600);
        assert_eq!(create["ResourceConfig"]["InstanceCount"], 1);
        assert_eq!(create["Tags"], json!([{"Key": "or_run", "Value": "r"}]));
    }

    #[tokio::test]
    async fn submit_refuses_a_region_without_a_default_image() {
        let Some(fake) = Fake::new().await else {
            crate::config::skip_without_tool("no Python 3.10+ on PATH");
            return;
        };
        let spec = json!({
            "name": "orx-r", "script": "x", "instanceType": "ml.m5.large", "env": {},
            "timeoutSeconds": 60, "volumeGb": 50, "roleArn": "arn", "s3Uri": "s3://bucket",
            "region": "af-south-1", "tags": {},
        });
        let out = fake.run(&["submit"], &spec.to_string(), &[]);
        let error = launcher_error(
            "submit",
            out.status.code(),
            &String::from_utf8_lossy(&out.stderr),
        );
        assert!(
            error.downcast_ref::<super::super::Refused>().is_some(),
            "{error}"
        );
        assert!(error.to_string().contains("--image"), "{error}");
        assert!(!events(&out).iter().any(|e| e.get("create").is_some()));
    }

    #[tokio::test]
    async fn status_maps_training_job_states() {
        let Some(fake) = Fake::new().await else {
            crate::config::skip_without_tool("no Python 3.10+ on PATH");
            return;
        };
        let status = |job: Value| {
            let out = fake.run(
                &["status", "orx-r", "sa-east-1"],
                "",
                &[("FAKE_JOB", &job.to_string())],
            );
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            serde_json::from_slice::<Value>(&out.stdout).unwrap()
        };
        assert_eq!(status(Value::Null), json!({"missing": true}));
        let queued = status(
            json!({"TrainingJobStatus": "InProgress", "SecondaryStatus": "Starting",
            "SecondaryStatusTransitions": [{"StatusMessage": "Preparing the instances"}]}),
        );
        assert_eq!(queued["stage"], "SCHEDULING");
        assert_eq!(queued["message"], "Preparing the instances");
        let running = status(
            json!({"TrainingJobStatus": "InProgress", "SecondaryStatus": "Training",
            "TrainingStartTime": 1700000000}),
        );
        assert_eq!(running["stage"], "RUNNING");
        assert_eq!(running["startedAt"], 1_700_000_000_000_i64);
        let done =
            status(json!({"TrainingJobStatus": "Completed", "SecondaryStatus": "Completed"}));
        assert_eq!(
            (done["stage"].as_str(), done["exitCode"].as_i64()),
            (Some("COMPLETED"), Some(0))
        );
        let failed = status(
            json!({"TrainingJobStatus": "Failed", "SecondaryStatus": "Failed",
            "FailureReason": "AlgorithmError: , exit code: 2"}),
        );
        assert_eq!(
            (failed["stage"].as_str(), failed["exitCode"].as_i64()),
            (Some("ERROR"), Some(2))
        );
        let timed_out = status(
            json!({"TrainingJobStatus": "Stopped", "SecondaryStatus": "MaxRuntimeExceeded",
            "StoppingCondition": {"MaxRuntimeInSeconds": 60}}),
        );
        assert_eq!(timed_out["stage"], "ERROR");
        assert!(
            timed_out["message"].as_str().unwrap().contains("60s"),
            "{timed_out}"
        );
        let stopped = status(json!({"TrainingJobStatus": "Stopped", "SecondaryStatus": "Stopped"}));
        assert_eq!(stopped["stage"], "CANCELED");
    }

    #[tokio::test]
    async fn cancel_stops_only_a_running_job_and_reports_a_missing_one() {
        let Some(fake) = Fake::new().await else {
            crate::config::skip_without_tool("no Python 3.10+ on PATH");
            return;
        };
        let cancel = |job: Value| {
            fake.run(
                &["cancel", "orx-r", "sa-east-1"],
                "",
                &[("FAKE_JOB", &job.to_string())],
            )
        };
        let out = cancel(json!({"TrainingJobStatus": "InProgress"}));
        assert_eq!(
            serde_json::from_slice::<Value>(&out.stdout).unwrap(),
            json!({"missing": false})
        );
        assert_eq!(events(&out), vec![json!({"stop": "orx-r"})]);
        let out = cancel(json!({"TrainingJobStatus": "Completed"}));
        assert!(events(&out).is_empty());
        let out = cancel(Value::Null);
        assert_eq!(
            serde_json::from_slice::<Value>(&out.stdout).unwrap(),
            json!({"missing": true})
        );
    }

    #[tokio::test]
    async fn logs_replay_from_the_start_and_end_with_the_job() {
        let Some(fake) = Fake::new().await else {
            crate::config::skip_without_tool("no Python 3.10+ on PATH");
            return;
        };
        let out = fake.run(
            &["logs", "orx-r", "sa-east-1"],
            "",
            &[("FAKE_JOB", r#"{"TrainingJobStatus": "Completed"}"#)],
        );
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&out.stdout), "step 1\nstep 2\n");
    }

    #[tokio::test]
    async fn missing_credentials_and_region_are_their_own_errors() {
        let Some(fake) = Fake::new().await else {
            crate::config::skip_without_tool("no Python 3.10+ on PATH");
            return;
        };
        let out = fake.run(&["status", "orx-r", ""], "", &[]);
        let error = launcher_error(
            "status",
            out.status.code(),
            &String::from_utf8_lossy(&out.stderr),
        );
        assert!(
            error.to_string().contains("ORX_SAGEMAKER_REGION"),
            "{error}"
        );
        let out = fake.run(
            &["status", "orx-r", ""],
            "",
            &[
                ("AWS_REGION", "eu-west-1"),
                ("FAKE_JOB", r#"{"TrainingJobStatus": "InProgress"}"#),
            ],
        );
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&out.stdout).unwrap()["region"],
            "eu-west-1"
        );
        let out = fake.run(
            &["status", "orx-r", "sa-east-1"],
            "",
            &[("FAKE_NO_CREDS", "1")],
        );
        let error = launcher_error(
            "status",
            out.status.code(),
            &String::from_utf8_lossy(&out.stderr),
        );
        assert!(error.to_string().contains("no AWS credentials"), "{error}");
    }

    #[tokio::test]
    async fn finish_deletes_only_the_env_file() {
        let Some(fake) = Fake::new().await else {
            crate::config::skip_without_tool("no Python 3.10+ on PATH");
            return;
        };
        let out = fake.run(
            &[
                "finish",
                "sa-east-1",
                "s3://bucket/research/orx/orx-r/input/",
            ],
            "",
            &[],
        );
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            events(&out),
            vec![json!({"delete": "research/orx/orx-r/input/env.sh"})]
        );
    }
}
