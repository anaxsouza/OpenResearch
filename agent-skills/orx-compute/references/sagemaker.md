# AWS SageMaker (`--backend sagemaker`)

Use this backend only when the user explicitly requests AWS SageMaker or it is
the configured default. Each run is one SageMaker training job on the user's
AWS account, billed there per second of instance time. SageMaker releases the
instance when the job ends, fails, reaches its timeout or is stopped, so nothing
keeps billing if orx quits.

Authentication uses the user's AWS credentials as the AWS SDK finds them:
`AWS_PROFILE` (including IAM Identity Center profiles after `aws sso login`, and
`aws login`), or
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. It also needs
`ORX_SAGEMAKER_ROLE_ARN`, the execution role the job runs as, and
`ORX_SAGEMAKER_S3_URI`, where runs stage their input and write their output.
`ORX_SAGEMAKER_REGION` picks the region; otherwise the profile's region is
used. orx saves these when the user connects AWS SageMaker, or reads them from
the environment. It never creates roles or buckets, and the job never receives
the caller's AWS credentials.

```sh
orx exp run <expId> --backend sagemaker --flavor ml.g5.xlarge
orx exp run <expId> --backend sagemaker --flavor ml.g6.2xlarge --timeout 12h
orx exp run <expId> --backend sagemaker --flavor ml.m5.large --image <ecr-image-uri>
```

- `--flavor` is required: a SageMaker instance type, with or without the `ml.`
  prefix. `ml.g4dn.xlarge` (T4, 16 GB), `ml.g5.xlarge` (A10G, 24 GB),
  `ml.g6.xlarge` (L4, 24 GB); CPU types such as `ml.m5.large`. The account's
  SageMaker training quota for that type and region must be at least 1.
- Timeout defaults to 4 hours and can be up to 28 days. It covers the whole
  job, including the minutes SageMaker spends downloading the image (about 4 on
  `ml.g4dn.xlarge` with the default image), so leave room for that. SageMaker
  stops the job at the limit, and the run fails as timed out.
- `--image` overrides the default AWS PyTorch Deep Learning Container (GPU or
  CPU by instance family). A custom image needs `bash` and `tar`; the default
  has no image for some regions, which then require `--image`.
- orx uploads the committed snapshot, the run script and the run's saved
  environment to `<ORX_SAGEMAKER_S3_URI>/orx/<job>/input/`; the env file is
  deleted when the run ends. Files the run writes to `/opt/ml/model` and
  `/opt/ml/output` are uploaded to `<ORX_SAGEMAKER_S3_URI>/orx/<job>/output/`.
- Logs come from CloudWatch Logs (`/aws/sagemaker/TrainingJobs`).

See [run monitoring](watching.md).
