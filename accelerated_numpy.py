import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# numpy interface
x = jnp.arange(10)
print(x)

# asynchronous dispatch
long_vector = jnp.arange(int(1e7))
jnp.dot(long_vector, long_vector).block_until_ready()


# grad
def sum_of_squares(x):
    return jnp.sum(x ** 2)


sum_of_squares_dx = jax.grad(sum_of_squares)
x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
print(sum_of_squares(x))
print(sum_of_squares_dx(x))


# grad on multiple variables
def sum_of_squares_error(x, y):
    return jnp.sum((x - y) ** 2)


sum_squared_error_dx = jax.grad(sum_of_squares_error)
y = jnp.asarray([1.1, 2.1, 3.1, 4.1])

print(sum_squared_error_dx(x, y))
print(jax.grad(sum_of_squares_error, argnums=(0, 1))(x, y))  # Find gradient wrt both x & y

# value and gradient
print(jax.value_and_grad(sum_of_squares_error)(x, y))


# return intermediary value
def squared_error_with_aux(x, y):
    return sum_of_squares_error(x, y), x - y


print(jax.grad(squared_error_with_aux, has_aux=True)(x, y))


# modifying arrays with pure functions
def jax_in_place_modify(x):
    return x.at[0].set(123)


y = jnp.array([1, 2, 3])
x = jax_in_place_modify(y)
print(x, y)

# linear regression example
xs = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))
ys = xs * 3 - 1 + noise
plt.scatter(xs, ys)


def model(theta, x):
    """Computes wx + b on a batch of input x."""
    w, b = theta
    return w * x + b


def loss_fn(theta, x, y):
    prediction = model(theta, x)
    return jnp.mean((prediction - y) ** 2)


theta = jnp.array([1., 1.])


@jax.jit
def update(theta, x, y, lr=0.1):
    return theta - lr * jax.grad(loss_fn)(theta, x, y)


for _ in range(1000):
    theta = update(theta, xs, ys)

plt.scatter(xs, ys)
plt.plot(xs, model(theta, xs))
plt.show()

w, b = theta
print(f"w: {w:<.2f}, b: {b:<.2f}")
