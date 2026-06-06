CFG = dict(
    # --- MODEL ARCHITECTURE ---                                        # --- DATA & TRAINING SYSTEM ---
    vocab_size      = 36864,                                            spm_prefix      = "genesis",
    block_size      = 2048,                                             data_split      = "train",
    layers          = 32,                                               batch_size      = 1,
    heads           = 12,                                               grad_accum      = 32,
    kv_heads        = 3,                                                dtype           = "float16",
    dim             = 1536,                                             num_workers     = 2,
    dropout         = 0.1,                                              prefetch_factor = 2,
    bias            = False,                                            chunk_size      = 64,
    grad_checkpoint = True,                                             shuffle_buffer  = 512,
    compile         = False,                                            

    # --- CHECKPOINT & LOG ---                                          # --- OPTIMIZER & SCHEDULER ---
    checkpoint_dir  = "genesis/checkpoints",                            total_steps     = 500_000_000,
    resume          = False,                                            warmup_steps    = 1000,
    save_every      = 1000,                                             lr              = 3e-4,
    log_every       = 25,                                               min_lr          = 1e-5,
    seed            = 55,                                               betas           = (0.9, 0.95),
                                                                        weight_decay    = 0.1,
                                                                        max_grad_norm   = 1.0,

                                                                        # --- HUGGINGFACE HUB ---
                                                                        hf_dataset_repo = "trucle5503/dataset_pretrain",
                                                                        hf_repo_id      = "trucle5503/Genesis",
                                                                        hf_token        = "",
)