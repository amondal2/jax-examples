import jax
import jax.numpy as jnp
from datetime import datetime

global_list = []


def log2(x):
    global_list.append(x)
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2


print(jax.make_jaxpr(log2)(3.0))


# just in time compilation example
def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


x = jnp.arange(1e6)
start = datetime.now()
selu(x).block_until_ready()
print(datetime.now() - start)

selu_jit = jax.jit(selu)
selu_jit(x).block_until_ready()
start = datetime.now()
selu_jit(x).block_until_ready()
print(datetime.now() - start)
