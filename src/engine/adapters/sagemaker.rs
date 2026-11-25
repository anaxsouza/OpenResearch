use super::api::{self, Api};
use crate::engine::*;
use crate::error::{anyhow, Result};
use crate::jobs::sagemaker;
use async_trait::async_trait;

pub struct SageMakerAccount;

struct Ids<'a> {
    job: &'a str,
    region: &'a str,
    input: &'a str,
}

fn ids(run: &WatchedRun) -> Result<Ids<'_>> {
    match &run.handle {
        Handle::Api { provider, ids, .. } if provider == "sagemaker_job" => match ids.as_slice() {
            [job, region, input] if !job.is_empty() && !region.is_empty() => Ok(Ids {
                job,
                region,
                input,
            }),
            _ => Err(anyhow!(
                "SageMaker handle is missing its training job name or region"
            )),
        },
        _ => Err(anyhow!("Run does not belong to AWS SageMaker")),
    }
}

fn unseen(job: &str, region: &str) -> String {
    format!(
        "SageMaker has no training job {job} in {region} for the AWS account orx signs in to now. \
         SageMaker never deletes a training job, so this doesn't prove it ended: orx is probably \
         signed in to another account. Check AWS_PROFILE, then the job in the SageMaker console."
    )
}

#[async_trait]
impl Api for SageMakerAccount {
    type Cursor = u64;

    async fn launch(&self, intent: &RunIntent, fence: &Fence<'_>) -> Result<Handle, LaunchError> {
        let mut spec: sagemaker::SageMakerJobSpec =
            serde_json::from_value(intent.launch.options.clone()).map_err(api::rejected)?;
        spec.name = sagemaker::job_name(&intent.run.id);
        spec.tags.insert("or_run".into(), intent.run.id.clone());
        spec.env
            .extend(crate::config::launch_env(&intent.run).map_err(api::rejected)?);
        sagemaker::strip_credentials(&mut spec.env);
        sagemaker::ensure_env().await.map_err(api::rejected)?;
        let submitted = sagemaker::run_job(&spec, Some(fence))
            .await
            .map_err(api::launch_error)?;
        Ok(api::handle(
            intent,
            vec![submitted.job_name, submitted.region, submitted.input_uri],
        ))
    }

    async fn recover(&self, intent: &RunIntent) -> Recovery {
        let name = sagemaker::job_name(&intent.run.id);
        let region = intent.launch.options["region"].as_str().unwrap_or("");
        match sagemaker::inspect_job(&name, region).await {
            Ok(Some(state)) => {
                let input = intent.launch.options["s3_uri"]
                    .as_str()
                    .map(|uri| {
                        format!("{}/orx/{name}/input/", uri.trim_end_matches('/'))
                    })
                    .unwrap_or_default();
                Recovery::Found(api::handle(intent, vec![name, state.region, input]))
            }
            Ok(None) => Recovery::Absent,
            Err(error) => api::unknown(format!(
                "Could not ask SageMaker for training job {name}: {error}"
            )),
        }
    }

    async fn inspect(&self, run: &WatchedRun) -> Result<Observation> {
        let ids = ids(run)?;
        let Some(state) = sagemaker::inspect_job(ids.job, ids.region).await? else {
            return Ok(Observation::Unreconciled {
                detail: unseen(ids.job, ids.region),
            });
        };
        Ok(api::stage(
            &state.stage,
            state.message,
            state.exit_code,
            state.started_at,
            state.finished_at,
        ))
    }

    async fn cancel(&self, run: &WatchedRun) -> Result<()> {
        let ids = ids(run)?;
        if sagemaker::cancel_job(ids.job, ids.region).await? {
            Ok(())
        } else {
            Err(anyhow!(unseen(ids.job, ids.region)))
        }
    }

    async fn finish(&self, run: &WatchedRun) -> Result<()> {
        let ids = ids(run)?;
        if ids.input.is_empty() {
            return Ok(());
        }
        sagemaker::finish_job(ids.region, ids.input).await
    }

    async fn logs(
        &self,
        run: &WatchedRun,
        cursor: &mut u64,
        window: std::time::Duration,
        sink: &mut (dyn for<'b, 'c> FnMut(&'b [u8], Option<&'c u64>) + Send),
    ) -> Result<bool> {
        let ids = ids(run)?;
        let mut received = 0;
        let outcome = {
            let mut lines = api::lines(sink, &mut received);
            let pass = sagemaker::stream_logs(
                ids.job,
                ids.region,
                *cursor,
                window.saturating_mul(2),
                &mut lines,
            );
            tokio::time::timeout(window, pass).await
        };
        let closed = api::closed(outcome, received)?;
        *cursor += received;
        Ok(closed.is_some())
    }
}
