import time
import pandas as pd
import disco.config as config
from disco.utils.trainer import ModelTrainer
from disco.utils.dataloader import AudioDataLoader
from disco.models.dense import DenseClassifier
from disco.utils.helpers import set_seeds, export_hyperparams


if __name__ == "__main__":
    start = time.time()
    set_seeds(config.SEED)
    print(f"==> Training classifiers on {config.DATASET_NAME} dataset ...")
    print(f"==> Using pretrained embeddings from {config.MODEL_NAME} ...")
    print(f"==> Augmentations: {config.AUGMENT} ...")
    dense_model = DenseClassifier(
        input_shape=config.MODELS.get(config.MODEL_NAME).get("in_dim"),
        num_classes=config.N_CLASSES,
        dropout=config.DROPOUT,
        l2_regularization=config.L2_REGULARIZATION,
    )
    trainer = ModelTrainer(
        dense_model, config.LEARNING_RATE, export_path=config.EXPORT_PATH
    )
    trainer.compile()

    print(f"==> Loading train and test splits from {config.PARTITIONS_PATH} ...")

    # get train, validation and test splits
    train_split = pd.read_csv(
        f"{config.PARTITIONS_PATH}/split01_train.csv", header=None, names=["filename"]
    )
    # randomly sample 20% of the train split for validation
    valid_split = train_split.sample(frac=0.2, random_state=config.SEED)
    train_split = train_split.drop(valid_split.index)  # remove val samples from train

    test_split = pd.read_csv(
        f"{config.PARTITIONS_PATH}/split01_test.csv",
        header=None,
        names=["filename"],
    )

    print("==> Preparing dataloaders ..")
    trainloader = AudioDataLoader(
        train_split["filename"].tolist(),
        batch_size=config.BATCH_SIZE,
        dataset="train",
        shuffle=True,
    )
    validloader = AudioDataLoader(
        valid_split["filename"].tolist(),
        batch_size=config.BATCH_SIZE,
        dataset="validation",
        shuffle=False,
    )
    testloader = AudioDataLoader(
        test_split["filename"].tolist(),
        batch_size=config.BATCH_SIZE,
        dataset="test",
        shuffle=False,
    )

    # generate train and test datasets
    train_dataset = trainloader.create_dataset()
    valid_dataset = validloader.create_dataset()
    test_dataset = testloader.create_dataset()

    # start training
    trainer.train(train_dataset, epochs=config.EPOCHS, validation_data=valid_dataset)

    # plot metrics
    trainer.plot_metrics()

    # evaluate on test set
    trainer.evaluate(test_dataset)

    # save the model
    trainer.save_model()

    # save the hyperparameters
    export_hyperparams(trainer.runid, f"{trainer.export_path}/hyperparams.json")

    print(f"==> Training took {(time.time() - start) / 60:.2f} mins ...")
