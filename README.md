# MLOps showcase

This repository includes code to showcase basic workflow for MLOps. Code is developed for AzureML, Machine Learning service running on Azure Cloud.

## Workflow
![Workflow](https://github.com/crayon/mlops-showcase/blob/main/images/workflow.png)

Presented workflow is very basic. It includes following steps:
* push your code changes to version control
* trigger GitHub Action workflow, which performs following
  * publishes AzureML pipeline
  * triggers AzureML pipeline

## AzureML Pipeline
Pipeline within AzureML is the core of showcase. It is a very simple ML pipeline that includes following steps:
* checks existing data available in Dataset and registers new version in case new data is specified
* prepares data for training
* model training using data from Dataset
* model evaluation and registration
* model deployment in case evaluation passed

## Code usage
Code is very simple to reuse. Prior to deployment, following has to be set up:
* Azure environment with Azure AD
* AzureML Workspace
* Azure AD Application with access to AzureML Workspace
* compute cluster and datastore within AzureML

For reuability of GitHub Actions, following secrets have to be available within GitHub repository:
* AUTH_RESOURCE_GROUP: resource group where AzureML workspace is deployed
* AUTH_SP_ID: Azure AD Application's application id
* AUTH_SP_SECRET: Azure AD Application's password
* AUTH_SUBSCR_ID: Azure subscription id
* AUTH_TENANT_ID: Azure AD tenant id
* AUTH_WORKSPACE_NAME: AzureML workspace name

### Comments
As mentioned, codebase is very basic with loads of posibilities for improvement. Reason for this is that the purpose of it is just to showcase some basic capabilities of AzureML and how to integrate very basic MLOps workflow into it.
