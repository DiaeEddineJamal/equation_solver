import tensorflow as tf
import numpy as np
from typing import Union, Tuple, Optional


class DifferentialSolver:
    def __init__(self, model: tf.keras.Model):
        """
        Initialize DifferentialSolver with a given neural network model.

        Args:
            model (tf.keras.Model): The physics-informed neural network model.
        """
        self.model = model

    def compute_gradients(
        self,
        x: tf.Tensor,
        t: Optional[tf.Tensor] = None,
        order: int = 2
    ) -> Union[Tuple[tf.Tensor, ...], tf.Tensor]:
        """
        Compute gradients for ODE or PDE based on input dimensions.

        Args:
            x (tf.Tensor): Spatial input tensor.
            t (tf.Tensor, optional): Time input tensor for PDEs. Defaults to None.
            order (int, optional): Maximum gradient order to compute. Defaults to 2.

        Returns:
            Union[Tuple[tf.Tensor, ...], tf.Tensor]: Computed gradients.
        """
        # Validate and preprocess input
        x = self._validate_input(x)
        if t is not None:
            t = self._validate_input(t)
            inputs = tf.concat([x, t], axis=1)
        else:
            inputs = x

        # Compute gradients using GradientTape
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            y_pred = self.model(inputs)
            gradients = self._compute_derivatives(tape, y_pred, x, t, order)

        # Clean up the persistent tape to avoid memory leaks
        del tape
        return gradients

    @staticmethod
    def _validate_input(tensor: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        """
        Validate and convert input to TensorFlow tensor.

        Args:
            tensor (Union[np.ndarray, tf.Tensor]): Input tensor.

        Returns:
            tf.Tensor: Validated and converted tensor.
        """
        if not isinstance(tensor, tf.Tensor):
            tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
        if len(tensor.shape) == 1:
            tensor = tf.expand_dims(tensor, axis=-1)
        return tensor

    def _compute_derivatives(
        self,
        tape: tf.GradientTape,
        y_pred: tf.Tensor,
        x: tf.Tensor,
        t: Optional[tf.Tensor] = None,
        order: int = 2
    ) -> Tuple[tf.Tensor, ...]:
        """
        Compute derivatives up to the specified order.

        Args:
            tape (tf.GradientTape): Gradient tape for derivative computation.
            y_pred (tf.Tensor): Model predictions.
            x (tf.Tensor): Spatial input.
            t (tf.Tensor, optional): Time input for PDEs.
            order (int): Maximum derivative order.

        Returns:
            Tuple[tf.Tensor, ...]: Computed derivatives.
        """
        derivatives = [y_pred]
        current_tensor = y_pred

        # Compute spatial derivatives
        for _ in range(order):
            derivative = tape.gradient(current_tensor, x)
            if derivative is None:
                raise ValueError("Gradient computation failed; ensure model supports differentiability.")
            derivatives.append(derivative)
            current_tensor = derivative

        # Compute time derivative for PDEs
        if t is not None:
            time_derivative = tape.gradient(y_pred, t)
            if time_derivative is None:
                raise ValueError("Time derivative computation failed.")
            derivatives.append(time_derivative)

        return tuple(derivatives)

    def solve_ode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Solve Ordinary Differential Equation (ODE).

        Args:
            x (tf.Tensor): Spatial input tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Prediction and first derivative.
        """
        gradients = self.compute_gradients(x)
        return gradients[0], gradients[1]

    def solve_pde(self, x: tf.Tensor, t: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        """
        Solve Partial Differential Equation (PDE).

        Args:
            x (tf.Tensor): Spatial input tensor.
            t (tf.Tensor): Time input tensor.

        Returns:
            Tuple[tf.Tensor, ...]: Prediction and derivatives.
        """
        return self.compute_gradients(x, t)

    def residual_loss(
        self,
        x: tf.Tensor,
        t: Optional[tf.Tensor] = None,
        target: Optional[tf.Tensor] = None,
        loss_fn: Optional[callable] = None
    ) -> tf.Tensor:
        """
        Compute residual loss for the differential equation.

        Args:
            x (tf.Tensor): Spatial input tensor.
            t (tf.Tensor, optional): Time input tensor for PDEs.
            target (tf.Tensor, optional): Target values for supervised loss.
            loss_fn (callable, optional): Custom loss function for residual computation.

        Returns:
            tf.Tensor: Computed residual loss.
        """
        gradients = self.compute_gradients(x, t)

        if loss_fn:
            return loss_fn(gradients, target)

        # Default ODE loss
        if t is None:
            _, first_derivative = gradients
            return tf.reduce_mean(tf.square(first_derivative))

        # Default PDE loss
        _, second_derivative, time_derivative = gradients
        if target is not None:
            return tf.reduce_mean(tf.square(time_derivative - second_derivative - target))
        return tf.reduce_mean(tf.square(time_derivative - second_derivative))
