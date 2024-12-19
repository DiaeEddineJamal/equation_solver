import tensorflow as tf
import numpy as np

class PINN(tf.keras.Model):
    def __init__(self,
                 layers,
                 equation_type,
                 activation='tanh',
                 initializer='glorot_uniform',
                 dropout_rate=0.0,
                 regularization=None):
        """
        Physics-Informed Neural Network Model

        Args:
            layers (list): Number of neurons in each hidden layer
            equation_type (str): Type of equation (ODE, PDE, etc.)
            activation (str, optional): Activation function. Defaults to 'tanh'.
            initializer (str, optional): Weight initialization method. Defaults to 'glorot_uniform'.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.0.
            regularization (dict, optional): Regularization configuration.
        """
        super(PINN, self).__init__()

        # Store configuration
        self.equation_type = equation_type
        self.layers_config = layers

        # Create regularizer
        self.regularizer = self._create_regularizer(regularization)

        # Build network layers
        self.hidden_layers = []
        for units in layers:
            layer = tf.keras.layers.Dense(
                units=units,
                activation=activation,
                kernel_initializer=initializer,
                kernel_regularizer=self.regularizer
            )
            self.hidden_layers.append(layer)

            # Optional dropout layer
            if dropout_rate > 0:
                dropout_layer = tf.keras.layers.Dropout(dropout_rate)
                self.hidden_layers.append(dropout_layer)

        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            1,
            kernel_initializer=initializer,
            kernel_regularizer=self.regularizer
        )

    def call(self, inputs, training=False):
        """
        Forward pass through the network

        Args:
            inputs (tf.Tensor): Input tensor
            training (bool, optional): Training mode flag. Defaults to False.

        Returns:
            tf.Tensor: Output tensor
        """
        x = inputs

        # Ensure inputs are of correct type and shape
        if not isinstance(inputs, tf.Tensor):
            x = tf.convert_to_tensor(inputs, dtype=tf.float32)

        # Reshape if needed
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=-1)

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)

        return self.output_layer(x)

    def _create_regularizer(self, regularization):
        """
        Create kernel regularizer based on configuration

        Args:
            regularization (dict): Regularization configuration

        Returns:
            tf.keras.regularizers: Regularizer instance
        """
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
        else:
            return None

    def get_config(self):
        """
        Get model configuration for serialization

        Returns:
            dict: Model configuration
        """
        config = super().get_config()
        config.update({
            "layers": self.layers_config,
            "equation_type": self.equation_type,
            "activation": self.hidden_layers[0].activation.__name__,
            "initializer": self.hidden_layers[0].kernel_initializer.__class__.__name__
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create model instance from configuration

        Args:
            config (dict): Model configuration

        Returns:
            PINN: Model instance
        """
        return cls(
            layers=config['layers'],
            equation_type=config['equation_type']
        )


### ODE Example: Solving dy/dx + y = 0 ###
def compute_loss(inputs, model):
    """
    Compute loss for the ODE: dy/dx + y = 0

    Args:
        inputs (tf.Tensor): Input data
        model (PINN): PINN model

    Returns:
        tf.Tensor: Loss value
    """
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
    grad_predictions = tape.gradient(predictions, inputs)
    residual = grad_predictions + predictions
    return tf.reduce_mean(tf.square(residual))


@tf.function
def train_step(inputs, model, optimizer):
    """
    Perform one training step

    Args:
        inputs (tf.Tensor): Input data
        model (PINN): PINN model
        optimizer (tf.keras.optimizers.Optimizer): Optimizer

    Returns:
        tf.Tensor: Loss value
    """
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
