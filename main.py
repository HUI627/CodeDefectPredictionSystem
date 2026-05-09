
        checkpoint = torch.load(config.model_save_dir / args.model_path,
                              map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        encoding = tokenizer(
            args.code,
            add_special_tokens=True,
            max_length=config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()

        logger.info(f"\nPrediction: {'Defect' if pred == 1 else 'No Defect'}")
        logger.info(f"Confidence: {probs[0][pred].item():.2%}")
        logger.info(f"Probabilities: No Defect={probs[0][0].item():.2%}, Defect={probs[0][1].item():.2%}")

if __name__ == '__main__':
    main()
