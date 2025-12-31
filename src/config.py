from dataclasses import dataclass, field

@dataclass
class HPARAMS:
    vocab_size = 14_000
    max_seq_len = 32
    batch_size = 64

    model_hparams: dict = field(default_factory=lambda: {
    "embedding_dim" : 512,
    "hidden_dim" : 512,
    "gru_layers" : 2,
    "gru_dropout" : 0.1,
    "pad_token_id" : 0
    })

    optimizer_hparams: dict = field(default_factory=lambda: {
        "lr": 1e-3,
        "weight_decay": 3e-5
    })


    trainer_hparams: dict = field(default_factory=lambda: {
    "n_epochs": 15,
    "enable_mixed_precision": True,
    "restore_best_model" : False,
    "use_early_stopping" : True,
    "early_stopping_patience" : 3,
    "grad_clip_value" : 1.0
    })
