import tensorflow as tf
import numpy as np

class PINNTrainer:
    def __init__(self, model, optimizer=None):
        """
        Initializes the PINN Trainer.

        Args:
            model (tf.keras.Model): The physics-informed neural network (PINN) model.
            optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer for training. Defaults to Adam optimizer.
        """
        self.model = model
        self.optimizer = optimizer if optimizer else tf.keras.optimizers.Adam(learning_rate=0.001)

    def _default_loss(self, y_true, y_pred):
        """
        Default loss function if not specified.

        Args:
            y_true (tf.Tensor): True values.
            y_pred (tf.Tensor): Predicted values.

        Returns:
            tf.Tensor: Computed loss.
        """
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def train_step_ode(self, training_data):
        """
        Training step for Ordinary Differential Equations (ODEs).

        Args:
            training_data (tf.Tensor): Input training data points.

        Returns:
            tf.Tensor: Total loss for the training step.
        """
        with tf.GradientTape() as tape:
            x = training_data
            y_pred = self.model(x)

            with tf.GradientTape() as tape_inner:
                tape_inner.watch(x)
                y = self.model(x)

            dy_dx = tape_inner.gradient(y, x)

            # Define the ODE loss
            ode_loss = self._compute_ode_loss(x, y, dy_dx)

            # Optional physics-informed loss
            physics_loss = self._default_loss(y_pred, y)

            # Combine losses
            total_loss = ode_loss + physics_loss

        # Compute gradients and apply them
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return total_loss

    def train_step_pde(self, x, t):
        """
        Training step for Partial Differential Equations (PDEs).

        Args:
            x (tf.Tensor): Spatial coordinates.
            t (tf.Tensor): Time coordinates.

        Returns:
            tf.Tensor: Total loss for the training step.
        """
        with tf.GradientTape(persistent=True) as tape:
            inputs = tf.concat([x, t], axis=1)
            u = self.model(inputs)

            # Compute derivatives
            with tf.GradientTape() as tape_inner:
                tape_inner.watch(inputs)
                u = self.model(inputs)

            du_dx = tape_inner.gradient(u, x)
            du_dt = tape_inner.gradient(u, t)
            d2u_dx2 = tape_inner.gradient(du_dx, x)

            # Define the PDE loss
            pde_loss = self._compute_pde_loss(u, du_dx, du_dt, d2u_dx2)

            # Optional boundary/initial condition loss
            bc_loss = self._default_loss(u, tf.zeros_like(u))

            # Combine losses
            total_loss = pde_loss + bc_loss

        # Compute gradients and apply them
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return total_loss

    def _compute_ode_loss(self, x, y, dy_dx):
        """
        Compute the loss for ODE.

        Args:
            x (tf.Tensor): Input coordinates.
            y (tf.Tensor): Predicted solution.
            dy_dx (tf.Tensor): Derivative of the solution.

        Returns:
            tf.Tensor: Computed ODE loss.
        """
        # Example ODE: dy/dx = -y
        ode_constraint = dy_dx + y
        return tf.reduce_mean(tf.square(ode_constraint))

    def _compute_pde_loss(self, u, du_dx, du_dt, d2u_dx2):
        """
        Compute the loss for PDE.

        Args:
            u (tf.Tensor): Predicted solution.
            du_dx (tf.Tensor): First spatial derivative.
            du_dt (tf.Tensor): Temporal derivative.
            d2u_dx2 (tf.Tensor): Second spatial derivative.

        Returns:
            tf.Tensor: Computed PDE loss.
        """
        # Example PDE: Heat equation (du/dt = α * d2u/dx2)
        alpha = 0.01  # Diffusion coefficient
        heat_equation_residual = du_dt - alpha * d2u_dx2
        return tf.reduce_mean(tf.square(heat_equation_residual))

    def train(self, dataset, epochs, mode="ode", verbose=True):
        """
        Trains the PINN for the specified number of epochs.

        Args:
            dataset (tf.data.Dataset): Dataset containing training data.
            epochs (int): Number of training epochs.
            mode (str): Training mode ("ode" or "pde").
            verbose (bool, optional): Whether to print progress. Defaults to True.

        Returns:
            None
        """
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataset:
                if mode == "ode":
                    total_loss += self.train_step_ode(batch)
                elif mode == "pde":
                    x, t = batch
                    total_loss += self.train_step_pde(x, t)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.numpy()}")
