import mlflow


class MLflowLogger:
    def __init__(self, tracking_uri: str, experiment_name: str, run_name=None):
        mlflow.set_tracking_uri(tracking_uri)
        self.run_name = run_name
        try:
            self.exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        except AttributeError:
            self.exp_id = mlflow.create_experiment(
                experiment_name,
                tags={"exp_type": "non-template"},
            )

    def log_metrics(self, metrics: dict, step: int = 0):
        mlflow.log_metrics(metrics, step)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_artifact(self, artifact_path: str, artifact_uri: str = None):
        mlflow.log_artifact(artifact_path, artifact_uri)

    def attach(self, func):
        def wrapper(*args, **kwargs):
            with mlflow.start_run(experiment_id=self.exp_id, run_name=self.run_name):
                func(*args, **kwargs)

        return wrapper

    def end_run(self, status: str = "FINISHED"):
        mlflow.end_run(status)

    def active_run(self):
        return mlflow.active_run()
