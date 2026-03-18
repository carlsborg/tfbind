from model_base import (
    ConvModel,
    ConvModel1Layer,
    ConvModel3Layer,
    ConvModelSmallKernel,
    ConvModelLargeKernel,
    ConvModelReLU,
    ConvModelSiLU,
    ConvModelLeakyReLU,
    ALL_MODELS,
)


class ModelFactory:
    """Factory for building models by id."""

    # Registry is built automatically from ALL_MODELS using each class's model_id().
    _registry = {cls().model_id(): cls for cls in ALL_MODELS}

    @staticmethod
    def build(model_id: str, **kwargs):
        """Build a model by its registered id.

        Args:
            model_id: Key in the registry (e.g. "conv").
            **kwargs: Forwarded to the model constructor.
        """
        if model_id not in ModelFactory._registry:
            raise ValueError(
                f"Unknown model id '{model_id}'. "
                f"Available: {list(ModelFactory._registry.keys())}"
            )
        return ModelFactory._registry[model_id](**kwargs)
