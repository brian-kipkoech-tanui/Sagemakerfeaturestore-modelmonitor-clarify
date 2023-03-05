# Monitoring a ML Workflow
ML monitoring should:

1. Capture events
2. Surface operational and ML-specific metrics
3. Emit alerts when events or metrics are outside of the expected range
4. Enable automation

## SageMaker Feature Store
Feature store is a database offered by AWS which helps developing features for machine learning inputs.

Feature store is useful because:

1. Multiple teams can use the same feature
2. Features can be monitored independently or in the context of a workflow
3. Surfaces insights about the feature
The code below shows you how to create a Feature Store FeatureGroup. To create it, you will first create the class (`FeatureGroup()` with its arguments `name` and `sagemaker_session`. Then you need to load the definitions, which can be done by passing a Panda's dataframe. Next, you can create the feature store by calling the `create` method.
```python
from sagemaker.feature_store.feature_group import FeatureGroup

feature_group = FeatureGroup(
    name='my-group', sagemaker_session=sagemaker_session)

feature_group.load_feature_definitions(data_frame=data)

feature_group.create(
    s3_uri=f"s3://{s3_bucket_name}/{prefix}",
    record_identifier_name=record_identifier,
    event_time_feature_name="EventTime",
    role_arn=role)
```
Creating a feature store doesn't mean there is data ready for use. You also need to ingest data to the datastore by calling the ingest method.
```python
feature_group.ingest(data_frame=test_data)
To get records from the feature store, you can use the SageMaker runtime client.

runtime = sagemaker_session.boto_session.client(
  'sagemaker-featurestore-runtime',
  region_name=region)

runtime.get_record(
    FeatureGroupName=feature_group_name,
    RecordIdentifierValueAsString=some_id_string)
```
## Monitoring ML Models with SageMaker Model monitor
Model Monitor will let you build trust in your ML systems because you can reason about its performance, and use alerts to quickly act on problems in the system.

One of the key objects is the `DataCaptureConfig` We provide this to the `deploy` function to capture events from the API (such as requests and responses).

### Configure model for monitoring
```python
capture_uri = f's3://{bucket}/data-capture'
data_capture_config = DataCaptureConfig(...)

xgb_predictor = model.deploy(
    ...
    data_capture_config=data_capture_config
)
```

### Define a Model Monitor class
```python
my_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)
```
Baselining is a powerful capability of Model Monitor that lets us automatically suggest guide rails for our model.
```python
my_monitor.suggest_baseline(
    baseline_dataset=s3_uri,
    dataset_format=DatasetFormat.csv(header=False),
)
```
Finally, we can schedule the monitoring job to run hourly (or daily, or any cron expression)
```python
my_monitor.create_monitoring_schedule(
    monitor_schedule_name=my_monitoring_schedule_name,
    endpoint_input=endpoint_name,
    statistics=my_monitor.baseline_statistics(),
    constraints=my_monitor.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly(),
)
```

## SageMaker Clarify
ageMaker Clarify in the Python SDK uses a very similar design to Model Monitor.

Clarify can help teams to ensure that your models are interacting with users in a responsible way and that their inferences are explainable.

We cover several of the Clarify functions and provide explanations about how we use them. Beginning with the `ModelExplainabilityMonitor`.
```python
ModelExplainabilityMonitor(
    role=role,
    sagemaker_session=session,
    max_runtime_in_seconds=timeout_duration,
)
```
Clarify comes with several of its own configuration functions, depending on what task you're using it for. We share an example, `SHAPConfig` for explainability analyses that use the SHAP algorithm.
```python
shap_config = sagemaker.clarify.SHAPConfig(
    ...
)
```
Clarify uses the same `create_monitoring_schedule` that is used in Model Monitor.

With these building blocks, you can check your model for biases or for explainability over time, and respond to alerts when it falls out of your acceptable thresholds.
