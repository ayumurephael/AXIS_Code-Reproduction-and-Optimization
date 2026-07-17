"""Full AXIS architecture runner using coherent counterfactual state objective v2."""
from .train_phase2_loss_redesign import main


if __name__ == "__main__":
    main(default_architecture_variant="full")
