
    def save_metrics(self, metrics):
        self.config.metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.metrics_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {self.config.metrics_dir / 'test_metrics.json'}")
