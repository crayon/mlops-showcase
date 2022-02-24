import os
from azureml.core import Experiment, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.pipeline.core import PublishedPipeline


auth_resource_group = os.environ.get('AUTH_RESOURCE_GROUP')
auth_sp_id = os.environ.get('AUTH_SP_ID')
auth_sp_secret = os.environ.get('AUTH_SP_SECRET')
auth_subscr_id = os.environ.get('AUTH_SUBSCR_ID')
auth_tenant_id = os.environ.get('AUTH_TENANT_ID')
auth_workspace_name = os.environ.get('AUTH_WORKSPACE_NAME')
build_id = os.environ.get('GITHUB_RUN_ID')
pipeline_name = "waicf-pipeline"

credentials = ServicePrincipalAuthentication(
    tenant_id=auth_tenant_id,
    service_principal_id=auth_sp_id,
    service_principal_password=auth_sp_secret
)
ws = Workspace(
    subscription_id=auth_subscr_id,
    resource_group=auth_resource_group,
    workspace_name=auth_workspace_name,
    auth=credentials
)

pipelines = PublishedPipeline.list(workspace=ws)
matched_pipelines = []

for p in pipelines:
    if p.name == pipeline_name and p.version == build_id:
        matched_pipelines.append(p)

if(len(matched_pipelines) > 1):
    published_pipeline = None
    raise Exception(f"Multiple active pipelines are published for build {build_id}.")
elif(len(matched_pipelines) == 0):
    published_pipeline = None
    raise KeyError(f"Unable to find a published pipeline for this build {build_id}")
else:
    published_pipeline = matched_pipelines[0]

    pipeline_run = Experiment(workspace=ws, name=pipeline_name).submit(published_pipeline, regenerate_outputs=False)

    print("Pipeline run initiated ", pipeline_run.id)
