CFG = dict(
    # --- MODEL & SYSTEM ---                                            # --- DATA, TRAINING & HF ---
    vocab_size      = 36864,                                            spm_prefix      = "genesis",
    block_size      = 2048,                                             data_split      = "train",
    layers          = 24,                                               batch_size      = 1,
    heads           = 16,                                               grad_accum      = 32,
    dim             = 1792,                                             num_workers     = 2,
    dropout         = 0.1,                                              prefetch_factor = 2,
    bias            = False,                                            chunk_size      = 64,
    grad_checkpoint = False,                                            shuffle_buffer  = 512,
    compile         = False,                                            dtype           = "float16",

    # --- CHECKPOINT & LOG ---                                          # --- OPTIMIZER & SCHEDULER ---
    checkpoint_dir  = "genesis/checkpoints",                            total_steps     = 100_000,
    resume          = False,                                            warmup_steps    = 1000,
    keep_last       = 3,                                                lr              = 3e-4,
    save_every      = 2,                                                min_lr          = 1e-5,
    log_every       = 10,                                               betas           = (0.9, 0.95),
    seed            = 55,                                               weight_decay    = 0.1,
                                                                        max_grad_norm   = 1.0,

    # --- HF HUB ---
    hf_dataset_repo = "trucle5503/dataset_pretrain",
    hf_repo_id      = "trucle5503/Genesis",
    hf_token        = '',
)