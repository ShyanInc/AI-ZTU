import numpy as np
import tensorflow as tf

n_samples, batch_size, num_steps = 1000, 100, 20000
learning_rate = 0.001

X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data)).shuffle(n_samples).batch(batch_size).repeat()
dataset_iter = iter(dataset)

class LinearModel(tf.Module):
    def __init__(self):
        self.k = tf.Variable(tf.random.normal((1, 1), dtype=tf.float32), name='slope')
        self.b = tf.Variable(tf.zeros((1,), dtype=tf.float32), name='bias')

    def __call__(self, X):
        return tf.matmul(X, self.k) + self.b

model = LinearModel()

@tf.function 
def compute_loss(X, y):
    y_pred = model(X)
    return tf.reduce_mean(tf.square(y - y_pred))

optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

@tf.function
def train_step(X_batch, y_batch):
    with tf.GradientTape() as tape:
        loss = compute_loss(X_batch, y_batch)
    gradients = tape.gradient(loss, [model.k, model.b])

    gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]

    optimizer.apply_gradients(zip(gradients, [model.k, model.b]))
    return loss

display_step = 100
for i in range(num_steps):
    X_batch, y_batch = next(dataset_iter)
    
    loss_val = train_step(X_batch, y_batch)

    if tf.math.is_nan(loss_val):
        print(f"NaN encountered in loss at iteration {i + 1}")
        break

    if (i + 1) % display_step == 0:
        print(f"Iteration {i + 1}: Loss={loss_val.numpy():.8f}, k={model.k.numpy()[0][0]:.4f}, b={model.b.numpy()[0]:.4f}")

print(f"Final parameters: k={model.k.numpy()[0][0]:.4f}, b={model.b.numpy()[0]:.4f}")
