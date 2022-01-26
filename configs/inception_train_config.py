inception_fm_net_config = {
    "data": {
        "input_size": (75, 75),
        "image_height": 75,
        "image_width": 75,
        "image_channel": 3,
    },
    "train": {
        "batch_size": 8,
        "epochs": 50,
        "learn_rate": 0.0001,
        "patience_learning_rate": 1,
        "factor_learning_rate": 0.1,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 8
    },
    "test": {
        "batch_size": 8,
        "F1_threshold": 0.5,
    }
}