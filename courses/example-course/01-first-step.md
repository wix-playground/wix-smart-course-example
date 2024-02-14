_**LAST UPDATED:** 26/9/2023, by [Ran Yahalom](https://wix.slack.com/archives/D028P8YJY64)_
<!-- TOC -->
* [What is the ML platform?](#what-is-the-ml-platform)
  * [**_Build_**: package your model for deployment](#build--package-your-model-for-deployment)
  * [**_Deploy_**: serve your model, so it's available for invocation](#deploy--serve-your-model-so-its-available-for-invocation)
  * [**_Trigger_**: invoke your model](#trigger--invoke-your-model)
  * [**_Monitor_**: continuously check your model's performance and health](#monitor--continuously-check-your-models-performance-and-health)
<!-- TOC -->

# What is the ML platform?

👉 The ML platform is a code framework developed in house to provide data scientists with the means to serve
(i.e. make available for use) and monitor their models.

👉 After you finish writing your model's code, you will usually rely on the ML platform to conduct the next stages of
your model's lifecycle as follows:

## **_Build_**: package your model for deployment

👉 Once you add your model's code to the Git repository, the ML platform will be triggered to package your model into a
deployable entity referred to as a "model build".

👉 You can also trigger a model build of a specific Git branch through the ML platform UI or programmatically via the ML
platform client library (aka the "Python SDK").

👉 You can have multiple model builds representing different versions of your model.

## **_Deploy_**: serve your model, so it's available for invocation

👉 Before a specific version of your model can be invoked through the ML platform, you need to make it available by
deploying the model build corresponding to that version.

👉 You can do this either manually through the ML platform UI or programmatically via the client library.

👉 When you need to expose your model for real-time invocations on relatively small sized prediction datasets, you deploy it as an **Online model**.

👉 When you need to expose your model for offline prediction (i.e. not in real-time) on larger datasets, you configure it as a "Batch prediction model" and it will be deployed by ML platform each time it is triggered.

## **_Trigger_**: invoke your model
👉 The ML platform makes it possible to trigger online models on provided prediction data in roughly three manners:
- Directly via the ML platform UI / client library.
- In reaction to real-time events you specify.
- By companies / services at Wix.

👉 Batch predictions can be triggered either through the ML platform UI or programmatically via the ML platform client library. 

👉 When a batch configured model is triggered, the prediction dataset is split into smaller batches on which the model is invoked simultaneously. 

👉 Key differences between Online vs. Batch deployments are:

|                                   | Online                                                                                                                                                                  | Batch                                                                                                                                                                                                                                                  |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Performance on **SMALL** datasets | 👍 FASTER because: <br/>- The model is instantly invoked when triggered (low latency).<br/>- Results are immediately returned.                                          | 👎 SLOWER because:<br/>- Usually requires at least 10-15 minutes after triggering until the model is actually invoked (high latency). <br/>- May require more time for additional post-invocation steps such as writing results to an output DB table. |
| Performance on **LARGE** datasets | 👎 May be infeasible to invoke on a large amount of data. Even if feasible:<br/>- May require large computational resources (e.g. RAM / CPU).<br/>- Might be very SLOW. | 👍 FASTER and computationally more efficient because the model is invoked on small subsets of the dataset simultaneously.                                                                                                                              |
| Cost                              | 👎 MORE EXPENSIVE because the required machine instances must be kept up and running 24/7.                                                                              | 👍  CHEAPER because the ML platform will only use the machine instances it needs on request and terminates them once the operation is done.                                                                                                            |

## **_Monitor_**: continuously check your model's performance and health

👉 The ML Platform provides integration with various tools to monitor and troubleshoot the performance of your model:

- Grafana Dashboards for Model's KPIs.
- Alerting mechanism.
- Grafana Deployment Logs.
- BI Events.
