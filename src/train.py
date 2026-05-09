
    def save_training_history(self):
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        self.config.metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.metrics_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    def load_model(self, filename):
        model_path = self.config.model_save_dir / filename
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
