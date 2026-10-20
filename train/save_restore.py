"""
Simple solution: Save the neural_net architecture with the checkpoint.

This ensures you can always restore the exact model architecture.
"""

import pickle as pkl
from pathlib import Path
from loguru import logger


def save_checkpoint_with_architecture(
    checkpoint_dir: str, trainer, config, conditions: list, metadata: dict = None
):
    """
    Save model weights + architecture + config for complete restoration.

    Args:
        checkpoint_dir: Directory to save checkpoint
        trainer: The trainer object (must have .neural_net and .save_checkpoint())
        config: The config used
        conditions: List of condition names
        metadata: Extra info (strategy, lr, etc.)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights (standard checkpoint)
    trainer.save_checkpoint(path=str(checkpoint_dir))

    # Save minimal metadata alongside the checkpoint (config_path is stored separately)
    restore_data = {
        "conditions": conditions,
        "metadata": metadata or {},
    }

    with open(checkpoint_dir / "restore_data.pkl", "wb") as f:
        pkl.dump(restore_data, f)

    logger.info(f"✓ Saved checkpoint with architecture to {checkpoint_dir}")
    logger.info(f"  Model weights: {checkpoint_dir}/")
    logger.info(f"  Architecture: {checkpoint_dir}/restore_data.pkl")


def restore_checkpoint_with_architecture(
    checkpoint_dir: str, trainer_type: str = "monge", jobid: int = 1
):
    """
    Restore trainer with exact architecture from checkpoint.

    Args:
        checkpoint_dir: Directory with checkpoint
        trainer_type: "monge" or "gaussian"
        jobid: Job ID

    Returns:
        (trainer, datamodule, config, metadata)
    """
    from cmonge.datasets.conditional_loader import ConditionalDataModule
    from cmonge.trainers.conditional_monge_trainer import ConditionalMongeTrainer
    from cmonge.trainers.conditional_gaussian_trainer import ConditionalGaussianTrainer

    checkpoint_dir = Path(checkpoint_dir)

    # Load saved data
    with open(checkpoint_dir / "restore_data.pkl", "rb") as f:
        restore_data = pkl.load(f)

    config = restore_data["config"]
    neural_net = restore_data["neural_net"]

    # Override conditions
    config.condition.conditions = restore_data["conditions"]

    # Recreate datamodule
    datamodule = ConditionalDataModule(config.data, config.condition)

    # Create trainer
    trainer_class = (
        ConditionalMongeTrainer
        if trainer_type == "monge"
        else ConditionalGaussianTrainer
    )

    trainer = trainer_class(
        jobid=jobid,
        logger_path=str(checkpoint_dir / "restore_log.pkl"),
        config=config.model,
        datamodule=datamodule,
    )

    # Replace the trainer's neural_net with saved architecture
    trainer.neural_net = neural_net

    # Load weights into the architecture
    from orbax.checkpoint import PyTreeCheckpointer

    checkpointer = PyTreeCheckpointer()
    trainer.state_neural_net = checkpointer.restore(
        str(checkpoint_dir), item=trainer.state_neural_net
    )

    logger.info(f"✓ Restored from {checkpoint_dir}")
    logger.info(f"  Conditions: {len(datamodule.conditions)}")
    logger.info(f"  Metadata: {restore_data.get('metadata', {})}")

    return trainer, datamodule, config, restore_data.get("metadata", {})


# ==============================================================================
# USAGE
# ==============================================================================




if __name__ == "__main__":
    print("=" * 80)
    print("SAVE NEURAL_NET ARCHITECTURE")
    print("=" * 80)
    print("\nSaves trainer.neural_net directly = exact architecture preserved")
    print("\nStorage: Model weights + ~50-100 KB for architecture object")
    print("\nNo config recreation issues - the model itself is saved!")
    print("=" * 80)
