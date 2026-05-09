
    def generate_all_plots(self, metrics, labels, probs):
        logger.info("Generating all visualizations...")
        self.plot_training_history()
        self.plot_confusion_matrix(np.array(metrics['confusion_matrix']))
        self.plot_roc_curve(labels, probs)
        self.plot_precision_recall_curve(labels, probs)
        self.plot_class_distribution(labels)
        logger.info("All visualizations generated successfully")
