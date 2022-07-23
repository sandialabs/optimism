from jax import config

# force double precision floating point arithmetic
# this is deprecated in jax, we'll have to find another way soon.
config.update("jax_enable_x64", True)

# silence warnings about no gpu/tpu
config.update("jax_platforms", "cpu")

# debugging options
#config.update("jax_debug_nans", True)
#config.update("jax_debug_infs", True)
#config.update("jax_disable_jit", True)

del config
