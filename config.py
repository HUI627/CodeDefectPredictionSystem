
        self.early_stopping_patience = 3
        self.save_best_model = True
        self.random_seed = 42

        self.device = "cuda"
        self.num_workers = 4

        self.log_interval = 10
        self.eval_interval = 100

config = Config()
