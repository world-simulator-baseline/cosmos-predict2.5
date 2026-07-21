from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (
    ActionChunkConditionedMinimalV1LVGDiT,
)


class RobotwinActionChunkConditionedDiT(ActionChunkConditionedMinimalV1LVGDiT):
    def init_weights(self):
        super().init_weights()

        for name in ("action_embedder_B_D", "action_embedder_B_3D"):
            action_encoder = getattr(self, name, None)
            if action_encoder is not None:
                action_encoder.fc1.reset_parameters()
                action_encoder.fc2.reset_parameters()