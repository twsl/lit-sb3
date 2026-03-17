from lit_sb3.policies.maskable import (
    MaskableActorCriticCnnPolicy,
    MaskableActorCriticPolicy,
    MaskableMultiInputActorCriticPolicy,
)

MlpPolicy = MaskableActorCriticPolicy
CnnPolicy = MaskableActorCriticCnnPolicy
MultiInputPolicy = MaskableMultiInputActorCriticPolicy
