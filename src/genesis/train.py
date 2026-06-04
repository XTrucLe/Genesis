import os
from dotenv import load_dotenv
from genesis.configs.cfg import CFG
# from genesis.core.trainer import Trainer
# from genesis.core.dataset import DataModule
# from genesis.core.checkpoint import CheckpointModule

load_dotenv()

def main():
    CFG["hf_token"] = os.getenv("HF_TOKEN", "")
    if not CFG["hf_token"]:
        raise ValueError("HF_TOKEN environment variable is not set.")
    
    print("Starting training with the following configuration:")
    for key, value in CFG.items():
        print(f"{key}: {value}")

    # data_manager = DataModule(CFG)
    # checkpoint_manager = CheckpointModule(CFG)
    # trainer = Trainer(CFG, data_manager, checkpoint_manager)
    # trainer.run()

if __name__ == "__main__":
    main()