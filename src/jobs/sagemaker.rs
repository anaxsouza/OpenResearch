//! AWS SageMaker Training Jobs backend — one training job per run, on the user's AWS account.
//!
//! Talks to AWS through the official Rust SDK, over the rustls-with-ring TLS the rest of orx
//! uses. `aws-config` resolves the standard credential chain (keys, `AWS_PROFILE`, IAM Identity
//! Center, `aws login`, instance roles); values saved in Settings → Environment are applied on
//! top, since they are not in orx's own environment. orx stores no AWS secret of its own.
//!
//! A training job releases its instance when it completes, fails, hits
//! `MaxRuntimeInSeconds` or is stopped, so nothing keeps billing if orx goes away. The
//! snapshot, the run script and the run's env file travel through one S3 input channel:
//! the job's `Environment` is readable by anyone who can describe it, and an entrypoint
//! item is capped at 256 characters. `finish` deletes the env file once the run is over.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_sdk_sagemaker::error::{DisplayErrorContext, ProvideErrorMetadata, SdkError};
use aws_sdk_sagemaker::operation::describe_training_job::DescribeTrainingJobOutput;
use aws_sdk_sagemaker::types as sm;

use crate::error::{anyhow, Result};

type HttpResponse = aws_sdk_sagemaker::config::http::HttpResponse;

pub const ROLE_ARN: &str = "ORX_SAGEMAKER_ROLE_ARN";
pub const S3_URI: &str = "ORX_SAGEMAKER_S3_URI";
pub const REGION: &str = "ORX_SAGEMAKER_REGION";
/// What the AWS credential chain reads; a saved value reaches orx's client, never the job.
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

const LOG_GROUP: &str = "/aws/sagemaker/TrainingJobs";
const CHANNEL: &str = "orx";
const INPUT: &str = "/opt/ml/input/data/orx";
const QUEUED: [&str; 6] = [
    "Starting",
    "Pending",
    "LaunchingMLInstances",
    "PreparingTrainingStack",
    "Downloading",
    "DownloadingTrainingImage",
];
/// Regions where AWS publishes its Deep Learning Containers from this account.
const DLC_ACCOUNT: &str = "763104351884";
const DLC_REGIONS: [&str; 17] = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ca-central-1",
    "sa-east-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-central-1",
    "eu-north-1",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
];
const GPU_FAMILIES: [&str; 11] = [
    "g4dn", "g5", "g6", "g6e", "p3", "p3dn", "p4d", "p4de", "p5", "p5e", "p5en",
];
const GPU_IMAGE: &str = "pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker";
const CPU_IMAGE: &str = "pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker";
const LOG_POLL: Duration = Duration::from_secs(2);

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

/// The AWS settings orx's client uses: the region given (a run's recorded one), else the
/// saved or inherited region; saved keys or profile over the inherited chain.
async fn sdk(region: Option<&str>) -> Result<SdkConfig> {
    use aws_smithy_http_client::tls::{rustls_provider::CryptoMode, Provider};
    let http = aws_smithy_http_client::Builder::new()
        .tls_provider(Provider::Rustls(CryptoMode::Ring))
        .build_https();
    let mut loader = aws_config::defaults(BehaviorVersion::latest()).http_client(http);
    let region = region
        .map(str::trim)
        .filter(|region| !region.is_empty())
        .map(str::to_string)
        .or_else(|| {
            [REGION, "AWS_REGION", "AWS_DEFAULT_REGION"]
                .iter()
                .find_map(|key| setting(key).map(|(value, _)| value))
        });
    if let Some(region) = region {
        loader = loader.region(Region::new(region));
    }
    match (
        setting("AWS_ACCESS_KEY_ID"),
        setting("AWS_SECRET_ACCESS_KEY"),
    ) {
        (Some((id, _)), Some((secret, _))) => {
            let token = setting("AWS_SESSION_TOKEN").map(|(value, _)| value);
            loader = loader.credentials_provider(aws_sdk_s3::config::Credentials::new(
                id, secret, token, None, "orx",
            ));
        }
        _ => match setting("AWS_PROFILE") {
            Some((profile, _)) => loader = loader.profile_name(profile),
            // Set but blank: unset to orx, but a profile named "" to the credential chain.
            None if std::env::var_os("AWS_PROFILE").is_some() => {
                loader = loader.profile_name("default")
            }
            None => {}
        },
    }
    let config = loader.load().await;
    if config.region().is_none() {
        return Err(super::Refused {
            status: None,
            message: format!(
                "No AWS region is set. Set {REGION} (or AWS_REGION), or a region in your AWS \
                 profile."
            ),
        }
        .into());
    }
    Ok(config)
}

fn region_of(config: &SdkConfig) -> String {
    config.region().map(ToString::to_string).unwrap_or_default()
}

/// Path-style S3 for an `AWS_ENDPOINT_URL` stand-in (LocalStack and the like), which serve
/// no per-bucket host names.
fn s3_client(config: &SdkConfig) -> aws_sdk_s3::Client {
    let path_style = setting("AWS_ENDPOINT_URL").is_some();
    aws_sdk_s3::Client::from_conf(
        aws_sdk_s3::config::Builder::from(config)
            .force_path_style(path_style)
            .build(),
    )
}

/// An AWS failure as one line the engine and the user can act on. A service error keeps its
/// code in parentheses, which is how needs-sign-in errors are recognised.
fn aws_error<E>(what: &str, err: SdkError<E, HttpResponse>) -> crate::error::Error
where
    E: ProvideErrorMetadata + std::error::Error + Send + Sync + 'static,
{
    if let SdkError::ServiceError(service) = &err {
        let error = service.err();
        return anyhow!(
            "{what} failed ({}): {}",
            error.code().unwrap_or("UnknownError"),
            error.message().unwrap_or("AWS gave no message")
        );
    }
    let detail = DisplayErrorContext(&err).to_string();
    if detail.to_ascii_lowercase().contains("credential") {
        anyhow!(
            "{what} failed: no AWS credentials could be loaded. Sign in (`aws sso login \
             --profile <name>` or `aws login`) and set AWS_PROFILE, or set AWS_ACCESS_KEY_ID and \
             AWS_SECRET_ACCESS_KEY, in the environment or Settings → Environment. ({detail})"
        )
    } else {
        anyhow!("{what} failed: could not reach AWS: {detail}")
    }
}

/// A service's refusal of a step nothing was created by: the engine fails the run on it.
fn refusal<E>(what: &str, err: SdkError<E, HttpResponse>) -> crate::error::Error
where
    E: ProvideErrorMetadata + std::error::Error + Send + Sync + 'static,
{
    match &err {
        SdkError::ServiceError(service) => super::Refused {
            status: Some(service.raw().status().as_u16()),
            message: format!(
                "{what}: {} ({})",
                service.err().message().unwrap_or("AWS gave no message"),
                service.err().code().unwrap_or("UnknownError")
            ),
        }
        .into(),
        _ => aws_error(what, err),
    }
}

fn is_missing<E: ProvideErrorMetadata>(err: &SdkError<E, HttpResponse>) -> bool {
    match err {
        SdkError::ServiceError(service) => {
            service.err().code() == Some("ValidationException")
                && service
                    .err()
                    .message()
                    .is_some_and(|m| m.to_ascii_lowercase().contains("not found"))
        }
        _ => false,
    }
}

fn split_s3(uri: &str) -> Result<(String, String)> {
    let rest = uri
        .strip_prefix("s3://")
        .ok_or_else(|| anyhow!("'{uri}' is not an s3://bucket/prefix URI."))?;
    let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
    Ok((bucket.to_string(), prefix.trim_matches('/').to_string()))
}

/// The AWS PyTorch Deep Learning Container for the region, GPU or CPU by instance family.
pub fn default_image(region: &str, instance_type: &str) -> Result<String> {
    if !DLC_REGIONS.contains(&region) {
        return Err(super::Refused {
            status: None,
            message: format!(
                "orx has no default image for region {region}. Pass --image with a training \
                 image this region can pull."
            ),
        }
        .into());
    }
    let family = instance_type.split('.').nth(1).unwrap_or("");
    let image = if GPU_FAMILIES.contains(&family) {
        GPU_IMAGE
    } else {
        CPU_IMAGE
    };
    Ok(format!(
        "{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/{image}"
    ))
}

fn shell_quote(value: &str) -> String {
    let safe = !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"@%+=:,./-_".contains(&b));
    if safe {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\"'\"'"))
    }
}

/// The env file the job sources: one `export` per variable, in a stable order.
pub fn env_file(env: &HashMap<String, String>) -> String {
    let mut keys: Vec<_> = env.keys().collect();
    keys.sort();
    keys.into_iter()
        .map(|key| format!("export {key}={}\n", shell_quote(&env[key])))
        .collect()
}

fn run_script(script: &str) -> String {
    format!("set -a; . {INPUT}/env.sh; set +a; mkdir -p /opt/orx; cd /opt/orx\n{script}")
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

pub struct Submitted {
    pub job_name: String,
    pub region: String,
    pub input_uri: String,
}

/// Stage the input channel and create the training job. A cancel accepted while staging
/// stops the launch before the job exists; recovery finds a job by its run-derived name.
pub async fn run_job(
    spec: &SageMakerJobSpec,
    fence: Option<&crate::engine::Fence<'_>>,
) -> Result<Submitted> {
    let config = sdk(spec.region.as_deref()).await?;
    let region = region_of(&config);
    let image = match spec.image.as_deref().map(str::trim) {
        Some(image) if !image.is_empty() => image.to_string(),
        _ => default_image(&region, &spec.instance_type)?,
    };
    let (bucket, prefix) = split_s3(&spec.s3_uri)?;
    let parent = if prefix.is_empty() {
        "orx".to_string()
    } else {
        format!("{prefix}/orx")
    };
    let base = format!("{parent}/{}", spec.name);
    let s3 = s3_client(&config);
    let put = |key: String, body: aws_sdk_s3::primitives::ByteStream| {
        s3.put_object().bucket(&bucket).key(key).body(body)
    };
    let staging = format!("Could not stage the run in s3://{bucket}/{base}");
    if let Some(archive) = &spec.source_archive {
        let body = aws_sdk_s3::primitives::ByteStream::from_path(archive)
            .await
            .map_err(|e| {
                anyhow!(
                    "Could not read the run's source archive {}: {e}",
                    archive.display()
                )
            })?;
        put(format!("{base}/input/source.tar"), body)
            .send()
            .await
            .map_err(|e| refusal(&staging, e))?;
        put(format!("{base}/input/source.tar.ready"), Vec::new().into())
            .send()
            .await
            .map_err(|e| refusal(&staging, e))?;
    }
    put(
        format!("{base}/input/env.sh"),
        env_file(&super::default_python_env(&spec.env))
            .into_bytes()
            .into(),
    )
    .server_side_encryption(aws_sdk_s3::types::ServerSideEncryption::Aes256)
    .send()
    .await
    .map_err(|e| refusal(&staging, e))?;
    put(
        format!("{base}/input/run.sh"),
        run_script(&spec.script).into_bytes().into(),
    )
    .send()
    .await
    .map_err(|e| refusal(&staging, e))?;
    if fence.is_some_and(|fence| fence.cancelled()) {
        return Err(anyhow!(
            "The run was cancelled before training job {} was created.",
            spec.name
        ));
    }

    let input_uri = format!("s3://{bucket}/{base}/input/");
    let algorithm = sm::AlgorithmSpecification::builder()
        .training_image(image)
        .training_input_mode(sm::TrainingInputMode::File)
        .container_entrypoint("bash")
        .container_entrypoint(format!("{INPUT}/run.sh"))
        .build();
    let source = sm::S3DataSource::builder()
        .s3_data_type(sm::S3DataType::S3Prefix)
        .s3_uri(&input_uri)
        .s3_data_distribution_type(sm::S3DataDistribution::FullyReplicated)
        .build();
    let channel = sm::Channel::builder()
        .channel_name(CHANNEL)
        .data_source(sm::DataSource::builder().s3_data_source(source).build())
        .build();
    // SageMaker appends <job>/output/ itself, so outputs land beside the run's input.
    let output = sm::OutputDataConfig::builder()
        .s3_output_path(format!("s3://{bucket}/{parent}"))
        .build();
    let resources = sm::ResourceConfig::builder()
        .instance_type(sm::TrainingInstanceType::from(spec.instance_type.as_str()))
        .instance_count(1)
        .volume_size_in_gb(i32::try_from(spec.volume_gb).unwrap_or(i32::MAX))
        .build();
    let mut tags: Vec<_> = spec.tags.iter().collect();
    tags.sort();
    let tags = tags
        .into_iter()
        .map(|(key, value)| sm::Tag::builder().key(key).value(value).build())
        .collect::<Vec<_>>();
    aws_sdk_sagemaker::Client::new(&config)
        .create_training_job()
        .training_job_name(&spec.name)
        .role_arn(&spec.role_arn)
        .algorithm_specification(algorithm)
        .input_data_config(channel)
        .output_data_config(output)
        .resource_config(resources)
        .stopping_condition(
            sm::StoppingCondition::builder()
                .max_runtime_in_seconds(i32::try_from(spec.timeout_seconds).unwrap_or(i32::MAX))
                .build(),
        )
        .set_tags(Some(tags))
        .send()
        .await
        .map_err(|e| {
            refusal(
                &format!("SageMaker didn't create training job {}", spec.name),
                e,
            )
        })?;
    Ok(Submitted {
        job_name: spec.name.clone(),
        region,
        input_uri,
    })
}

#[derive(Debug, Clone)]
pub struct JobState {
    pub stage: String,
    pub message: Option<String>,
    pub exit_code: Option<i32>,
    pub region: String,
    pub started_at: Option<i64>,
    pub finished_at: Option<i64>,
}

fn exit_code(reason: &str) -> Option<i32> {
    let lower = reason.to_ascii_lowercase();
    let after = &reason[lower.find("exit code")? + "exit code".len()..];
    let digits: String = after
        .trim_start_matches([':', ' '])
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// A described training job in the shared stage vocabulary.
pub fn job_state(job: &DescribeTrainingJobOutput, region: &str) -> JobState {
    let millis = |at: Option<&aws_sdk_sagemaker::primitives::DateTime>| {
        at.and_then(|at| at.to_millis().ok())
    };
    let secondary = job.secondary_status().map(|s| s.as_str()).unwrap_or("");
    let mut state = JobState {
        stage: "RUNNING".into(),
        message: None,
        exit_code: None,
        region: region.to_string(),
        started_at: millis(job.training_start_time()),
        finished_at: millis(job.training_end_time()),
    };
    match job.training_job_status() {
        Some(sm::TrainingJobStatus::Completed) => {
            state.stage = "COMPLETED".into();
            state.exit_code = Some(0);
        }
        Some(sm::TrainingJobStatus::Failed) => {
            let reason = job.failure_reason().unwrap_or("the training job failed");
            state.stage = "ERROR".into();
            state.exit_code = exit_code(reason);
            state.message = Some(reason.to_string());
        }
        Some(sm::TrainingJobStatus::Stopped) if secondary == "MaxRuntimeExceeded" => {
            let limit = job
                .stopping_condition()
                .and_then(|c| c.max_runtime_in_seconds())
                .unwrap_or_default();
            state.stage = "ERROR".into();
            state.message = Some(format!(
                "timed out: SageMaker stopped the job at its {limit}s limit"
            ));
        }
        Some(sm::TrainingJobStatus::Stopped) => state.stage = "CANCELED".into(),
        Some(sm::TrainingJobStatus::InProgress) if QUEUED.contains(&secondary) => {
            state.stage = "SCHEDULING".into();
            state.message = job
                .secondary_status_transitions()
                .last()
                .and_then(|t| t.status_message())
                .map(str::to_string)
                .or_else(|| Some(secondary.to_string()));
        }
        _ => {}
    }
    state
}

/// `None` when SageMaker has no job by that name in that region. An empty region is the
/// profile's own, as at submit.
pub async fn inspect_job(name: &str, region: &str) -> Result<Option<JobState>> {
    let config = sdk(Some(region)).await?;
    match aws_sdk_sagemaker::Client::new(&config)
        .describe_training_job()
        .training_job_name(name)
        .send()
        .await
    {
        Ok(job) => Ok(Some(job_state(&job, &region_of(&config)))),
        Err(err) if is_missing(&err) => Ok(None),
        Err(err) => Err(aws_error("DescribeTrainingJob", err)),
    }
}

/// False when SageMaker has no job by that name in that region.
pub async fn cancel_job(name: &str, region: &str) -> Result<bool> {
    let config = sdk(Some(region)).await?;
    let client = aws_sdk_sagemaker::Client::new(&config);
    let job = match client
        .describe_training_job()
        .training_job_name(name)
        .send()
        .await
    {
        Ok(job) => job,
        Err(err) if is_missing(&err) => return Ok(false),
        Err(err) => return Err(aws_error("DescribeTrainingJob", err)),
    };
    if job.training_job_status() == Some(&sm::TrainingJobStatus::InProgress) {
        client
            .stop_training_job()
            .training_job_name(name)
            .send()
            .await
            .map_err(|err| aws_error("StopTrainingJob", err))?;
    }
    Ok(true)
}

pub async fn finish_job(region: &str, input_uri: &str) -> Result<()> {
    let config = sdk(Some(region)).await?;
    let (bucket, prefix) = split_s3(input_uri)?;
    s3_client(&config)
        .delete_object()
        .bucket(bucket)
        .key(format!("{prefix}/env.sh"))
        .send()
        .await
        .map_err(|err| aws_error("Deleting the run's env file", err))?;
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
    let sdk_config = sdk(config.region.as_deref()).await?;
    let identity = aws_sdk_sts::Client::new(&sdk_config)
        .get_caller_identity()
        .send()
        .await
        .map_err(|err| aws_error("GetCallerIdentity", err))?;
    let (bucket, _) = split_s3(&config.s3_uri)?;
    s3_client(&sdk_config)
        .head_bucket()
        .bucket(&bucket)
        .send()
        .await
        .map_err(|err| match &err {
            SdkError::ServiceError(service) => anyhow!(
                "Can't reach S3 bucket {bucket} with these credentials (HTTP {}).",
                service.raw().status().as_u16()
            ),
            _ => aws_error("HeadBucket", err),
        })?;
    Ok(Identity {
        account: identity.account().unwrap_or_default().to_string(),
        arn: identity.arn().unwrap_or_default().to_string(),
        region: region_of(&sdk_config),
    })
}

/// One pass over the job's CloudWatch log, invoking `sink` per line past `skip`. Each pass
/// replays the log from its first line; ends once the job is over and the log has gone
/// quiet, or after `idle` without a new line.
pub async fn stream_logs(
    name: &str,
    region: &str,
    skip: u64,
    idle: Duration,
    sink: &mut (dyn FnMut(&str) + Send),
) -> Result<u64> {
    let config = sdk(Some(region)).await?;
    let logs = aws_sdk_cloudwatchlogs::Client::new(&config);
    let sagemaker = aws_sdk_sagemaker::Client::new(&config);
    let mut tokens: HashMap<String, String> = HashMap::new();
    let mut seen = 0u64;
    let mut quiet_after_end = 0;
    let mut last_line = Instant::now();
    loop {
        let streams = match logs
            .describe_log_streams()
            .log_group_name(LOG_GROUP)
            .log_stream_name_prefix(format!("{name}/"))
            .send()
            .await
        {
            Ok(out) => {
                let mut names: Vec<String> = out
                    .log_streams()
                    .iter()
                    .filter_map(|s| s.log_stream_name().map(str::to_string))
                    .collect();
                names.sort();
                names
            }
            Err(SdkError::ServiceError(service))
                if service.err().code() == Some("ResourceNotFoundException") =>
            {
                Vec::new()
            }
            Err(err) => return Err(aws_error("DescribeLogStreams", err)),
        };
        let mut got = false;
        for stream in streams {
            loop {
                let token = tokens.get(&stream).cloned();
                let page = logs
                    .get_log_events()
                    .log_group_name(LOG_GROUP)
                    .log_stream_name(&stream)
                    .start_from_head(true)
                    .set_next_token(token.clone())
                    .send()
                    .await
                    .map_err(|err| aws_error("GetLogEvents", err))?;
                for event in page.events() {
                    for line in event
                        .message()
                        .unwrap_or("")
                        .trim_end_matches('\n')
                        .split('\n')
                    {
                        got = true;
                        seen += 1;
                        if seen > skip {
                            sink(line);
                        }
                    }
                }
                let next = page.next_forward_token().map(str::to_string);
                let done = next.is_none() || next == token;
                if let Some(next) = next {
                    tokens.insert(stream.clone(), next);
                }
                if done {
                    break;
                }
            }
        }
        if got {
            last_line = Instant::now();
            quiet_after_end = 0;
        } else {
            let over = match sagemaker
                .describe_training_job()
                .training_job_name(name)
                .send()
                .await
            {
                Ok(job) => {
                    job_state(&job, region).stage != "RUNNING"
                        && job_state(&job, region).stage != "SCHEDULING"
                }
                Err(err) if is_missing(&err) => true,
                Err(err) => return Err(aws_error("DescribeTrainingJob", err)),
            };
            if over {
                quiet_after_end += 1;
                if quiet_after_end > 1 {
                    break;
                }
            } else if last_line.elapsed() >= idle {
                break;
            }
        }
        tokio::time::sleep(crate::config::paced(LOG_POLL)).await;
    }
    Ok(seen.max(skip))
}

/// Fail fast before staging if SageMaker isn't usable: settings, credentials, bucket.
pub async fn preflight() -> Result<()> {
    check(&config()?).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            split_s3("s3://bucket/research/").unwrap(),
            ("bucket".to_string(), "research".to_string())
        );
        assert_eq!(
            split_s3("s3://bucket").unwrap(),
            ("bucket".to_string(), String::new())
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
    fn env_file_quotes_every_value_for_bash() {
        let env = HashMap::from([
            ("HF_TOKEN".to_string(), "s3cr'et".to_string()),
            ("A".to_string(), "plain-1.0".to_string()),
            ("B".to_string(), "two words".to_string()),
            ("C".to_string(), String::new()),
        ]);
        assert_eq!(
            env_file(&env),
            "export A=plain-1.0\nexport B='two words'\nexport C=''\nexport HF_TOKEN='s3cr'\"'\"'et'\n"
        );
    }

    #[test]
    fn default_images_follow_region_and_instance_family() {
        assert_eq!(
            default_image("sa-east-1", "ml.g4dn.xlarge").unwrap(),
            format!("763104351884.dkr.ecr.sa-east-1.amazonaws.com/{GPU_IMAGE}")
        );
        assert!(default_image("us-east-1", "ml.m5.large")
            .unwrap()
            .ends_with(CPU_IMAGE));
        let refused = default_image("af-south-1", "ml.g5.xlarge").unwrap_err();
        assert!(refused.downcast_ref::<super::super::Refused>().is_some());
        assert!(refused.to_string().contains("--image"), "{refused}");
    }

    fn described(status: sm::TrainingJobStatus, secondary: &str) -> aws_sdk_sagemaker::operation::describe_training_job::builders::DescribeTrainingJobOutputBuilder{
        DescribeTrainingJobOutput::builder()
            .training_job_status(status)
            .secondary_status(sm::SecondaryStatus::from(secondary))
    }

    #[test]
    fn training_job_states_map_to_run_stages() {
        let state = |b: aws_sdk_sagemaker::operation::describe_training_job::builders::DescribeTrainingJobOutputBuilder| job_state(&b.build(), "sa-east-1");
        let queued = state(
            described(sm::TrainingJobStatus::InProgress, "Starting").secondary_status_transitions(
                sm::SecondaryStatusTransition::builder()
                    .status(sm::SecondaryStatus::Starting)
                    .status_message("Preparing the instances")
                    .start_time(aws_sdk_sagemaker::primitives::DateTime::from_secs(0))
                    .build(),
            ),
        );
        assert_eq!(
            (queued.stage.as_str(), queued.message.as_deref()),
            ("SCHEDULING", Some("Preparing the instances"))
        );
        let running = state(
            described(sm::TrainingJobStatus::InProgress, "Training").training_start_time(
                aws_sdk_sagemaker::primitives::DateTime::from_secs(1_700_000_000),
            ),
        );
        assert_eq!(
            (running.stage.as_str(), running.started_at),
            ("RUNNING", Some(1_700_000_000_000))
        );
        assert_eq!(running.region, "sa-east-1");
        let done = state(described(sm::TrainingJobStatus::Completed, "Completed"));
        assert_eq!(
            (done.stage.as_str(), done.exit_code),
            ("COMPLETED", Some(0))
        );
        let failed = state(
            described(sm::TrainingJobStatus::Failed, "Failed")
                .failure_reason("AlgorithmError: , exit code: 2"),
        );
        assert_eq!(
            (failed.stage.as_str(), failed.exit_code),
            ("ERROR", Some(2))
        );
        let unknown_exit = state(
            described(sm::TrainingJobStatus::Failed, "Failed")
                .failure_reason("CapacityError: no capacity"),
        );
        assert_eq!(
            (unknown_exit.exit_code, unknown_exit.message.as_deref()),
            (None, Some("CapacityError: no capacity"))
        );
        let timed_out = state(
            described(sm::TrainingJobStatus::Stopped, "MaxRuntimeExceeded").stopping_condition(
                sm::StoppingCondition::builder()
                    .max_runtime_in_seconds(60)
                    .build(),
            ),
        );
        assert_eq!(timed_out.stage, "ERROR");
        assert!(timed_out.message.unwrap().contains("60s"));
        assert_eq!(
            state(described(sm::TrainingJobStatus::Stopped, "Stopped")).stage,
            "CANCELED"
        );
    }
}
