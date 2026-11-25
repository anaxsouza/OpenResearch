//! AWS SageMaker launch intents. The run's env is added only at launch.

use std::collections::HashMap;

use crate::compute::SourceSnapshot;
use crate::error::Result;
use crate::jobs::{huggingface as hf, sagemaker};
use crate::store::LaunchSpec;

const ARCHIVE: &str = "/opt/ml/input/data/orx/source.tar";

pub fn launch_spec(
    args: &crate::ExpRunArgs,
    source: &SourceSnapshot,
    run_id: &str,
    command: &str,
) -> Result<LaunchSpec> {
    let instance_type = sagemaker::resolve_flavor(args.flavor.as_deref())?;
    let timeout_seconds = match &args.timeout {
        Some(t) => hf::parse_timeout(t)?,
        None => sagemaker::DEFAULT_TIMEOUT_SECS,
    };
    sagemaker::check_timeout(timeout_seconds)?;
    let config = sagemaker::config()?;
    let spec = sagemaker::SageMakerJobSpec {
        name: sagemaker::job_name(run_id),
        script: crate::compute::gated_script(ARCHIVE, command),
        image: args.image.clone(),
        instance_type,
        env: HashMap::new(),
        timeout_seconds,
        volume_gb: sagemaker::VOLUME_GB,
        role_arn: config.role_arn,
        s3_uri: config.s3_uri,
        region: config.region,
        tags: super::run_labels(&args.exp_id, run_id)?,
        source_archive: Some(source.path.clone()),
    };
    Ok(LaunchSpec {
        kind: "sagemaker_job".into(),
        target: None,
        options: serde_json::to_value(spec)?,
    })
}
