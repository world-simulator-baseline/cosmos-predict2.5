import torch.nn as nn

from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (
    ActionChunkConditionedMinimalV1LVGDiT,
)


class RobotwinActionChunkConditionedDiT(ActionChunkConditionedMinimalV1LVGDiT):
    """Keep the pretrained DiT unchanged while adding a safe action residual path.

    The base constructor first calls this hook before it creates the action
    embedders, then the training model calls it again after materializing the
    meta-device network with ``to_empty``.  The ``getattr`` guard makes the
    first call a no-op and applies this initialization on the second call.
    """

    def init_weights(self):
        super().init_weights()
        for name in ("action_embedder_B_D", "action_embedder_B_3D"):
            action_encoder = getattr(self, name, None)
            if action_encoder is None:
                continue
            # A random fc1 gives fc2 a non-zero hidden activation and gradient
            # on the first update.  A zero fc2 makes the initial action residual
            # exactly zero, preserving the pretrained DiT's initial behavior.
            action_encoder.fc1.reset_parameters()
            nn.init.zeros_(action_encoder.fc2.weight)
            nn.init.zeros_(action_encoder.fc2.bias)
