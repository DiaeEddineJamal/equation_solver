import tensorflow as tf
import numpy as np

class PINN(tf.keras.Model):
    def __init__(self, layers, equation_type, activation='tanh',
                 initializer='glorot_uniform', dropout_rate=0.0, regularization=None):
        super(PINN, self).__init__()

        # Store configuration
        self.equation_type = equation_type
        self.layers_config = layers
        self.activation = activation
        self.initializer = initializer

        # Regularizer creation
        self.regularizer = self._create_regularizer(regularization)

        # Build network layers
        self.hidden_layers = []
        for units in layers:
            self.hidden_layers.append(tf.keras.layers.Dense(
                units=units,
                activation=tf.keras.activations.get(activation),
                kernel_initializer=tf.keras.initializers.get(initializer),
                kernel_regularizer=self.regularizer
            ))
            if dropout_rate > 0:
                self.hidden_layers.append(tf.keras.layers.Dropout(rate=dropout_rate))

        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.get(initializer),
            kernel_regularizer=self.regularizer
        )

    def call(self, inputs, training=False):
        # Ensure inputs are tensors with the correct shape
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=-1)

        for layer in self.hidden_layers:
            x = layer(x, training=training)

        return self.output_layer(x)

    def _create_regularizer(self, regularization):
        if not regularization:
            return None
        reg_type = regularization.get('type', 'l2')
        reg_scale = regularization.get('scale', 0.001)
        if reg_type == 'l1':
            return tf.keras.regularizers.l1(reg_scale)
        elif reg_type == 'l2':
            return tf.keras.regularizers.l2(reg_scale)
        elif reg_type == 'l1_l2':
            return tf.keras.regularizers.l1_l2(l1=reg_scale, l2=reg_scale)
        return None

    def get_config(self):
        config = super().get_config()
        config.update({
            "layers": self.layers_config,
            "equation_type": self.equation_type,
            "activation": self.activation,
            "initializer": self.initializer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            layers=config['layers'],
            equation_type=config['equation_type'],
            activation=config.get('activation', 'tanh'),
            initializer=config.get('initializer', 'glorot_uniform')
        )


def compute_loss(inputs, model):
    # Compute residuals for the ODE: dy/dx + y = 0
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
    grad_predictions = tape.gradient(predictions, inputs)
    residual = grad_predictions + predictions
    return tf.reduce_mean(tf.square(residual))


@tf.function
def train_step(inputs, model, optimizer):
    # Training step with gradient computation and application
    with tf.GradientTape() as tape:
        loss = compute_loss(inputs, model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Define training data
x_data = tf.convert_to_tensor(np.linspace(0, 1, 100), dtype=tf.float32)

# Initialize the model
model = PINN(layers=[10, 10], equation_type="ODE")

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Train the model
epochs = 1000
for epoch in range(epochs):
    loss = train_step(x_data, model, optimizer)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy()}")
