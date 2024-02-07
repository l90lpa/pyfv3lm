from jax._src.effects import (
        Effect,
        lowerable_effects,
        ordered_effects,
        control_flow_allowed_effects,
        custom_derivatives_allowed_effects,
    )

class OrderedMPPEffect(Effect):
    def __hash__(self):
        return int(39673729008)

ordered_effect = OrderedMPPEffect()

lowerable_effects.add_type(OrderedMPPEffect)
ordered_effects.add_type(OrderedMPPEffect)
control_flow_allowed_effects.add_type(OrderedMPPEffect)
custom_derivatives_allowed_effects.add_type(OrderedMPPEffect)
