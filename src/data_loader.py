
    def _save_processed_data(self, train_data, val_data, test_data, stats):
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(train_data).to_csv(self.processed_data_dir / 'train.csv', index=False)
        pd.DataFrame(val_data).to_csv(self.processed_data_dir / 'val.csv', index=False)
        pd.DataFrame(test_data).to_csv(self.processed_data_dir / 'test.csv', index=False)

        with open(self.processed_data_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Processed data saved to {self.processed_data_dir}")
        logger.info(f"Dataset statistics: {stats}")

    def load_processed_data(self):
        train_df = pd.read_csv(self.processed_data_dir / 'train.csv')
        val_df = pd.read_csv(self.processed_data_dir / 'val.csv')
        test_df = pd.read_csv(self.processed_data_dir / 'test.csv')

        with open(self.processed_data_dir / 'stats.json', 'r') as f:
            stats = json.load(f)

        return train_df, val_df, test_df, stats
