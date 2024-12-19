import json
from flask import Flask, request, jsonify
from .model import PINN
from .solver import DifferentialSolver
from .train import PINNTrainer
from .utils import create_training_data
import numpy as np
import tensorflow as tf

class PINNService:
    def __init__(self):
        self.model = None
        self.solver = None
        self.trainer = None

    def initialize_model(self, config):
        """
        Initialize the PINN model with given configuration

        Args:
            config (dict): Configuration parameters for model initialization

        Returns:
            dict: Initialization status
        """
        try:
            self.model = PINN(
                layers=config['layers'],
                equation_type=config['equation_type']
            )
            self.solver = DifferentialSolver(self.model)
            self.trainer = PINNTrainer(self.model, self.solver)

            return {
                'status': 'success',
                'message': 'Model initialized successfully',
                'details': {
                    'equation_type': config['equation_type'],
                    'layers': config['layers']
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def solve_equation(self, data):
        """
        Solve differential equation using PINN

        Args:
            data (dict): Solving configuration and parameters

        Returns:
            dict: Solution results
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")

        try:
            equation_type = data['equation_type']
            domain = data['domain']
            num_points = data['num_points']

            # Load example data if provided
            example_id = data.get('example_id')
            if example_id:
                try:
                    with open(f'data/examples/{equation_type.lower()}_example_{example_id}.json', 'r') as f:
                        example_data = json.load(f)
                        # Update parameters based on example
                        domain = example_data.get('domain', domain)
                        num_points = example_data.get('num_points', num_points)
                except FileNotFoundError:
                    print(f"Example {example_id} not found. Using provided parameters.")

            # Create training data
            training_data = create_training_data(equation_type, domain, num_points)

            # Training process
            losses = []
            epochs = data.get('epochs', 1000)
            for epoch in range(epochs):
                try:
                    if equation_type == "ODE":
                        loss = float(self.trainer.train_step_ode(training_data))
                    else:
                        x, t = tf.split(training_data, 2, axis=1)
                        loss = float(self.trainer.train_step_pde(x, t))

                    losses.append(loss)
                except Exception as train_error:
                    print(f"Training error at epoch {epoch}: {train_error}")
                    break

            # Generate solution
            solution = self._generate_solution(equation_type, domain)

            # Save results if requested
            if data.get('save_result'):
                result_data = {
                    'losses': losses,
                    'solution': solution,
                    'parameters': data
                }
                result_id = data.get('result_id', 'default')
                with open(f'data/results/{equation_type.lower()}_result_{result_id}.json', 'w') as f:
                    json.dump(result_data, f)

            return {
                'status': 'success',
                'losses': losses,
                'solution': solution
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def _generate_solution(self, equation_type, domain):
        """
        Generate solution based on equation type

        Args:
            equation_type (str): Type of equation (ODE or PDE)
            domain (list/tuple): Domain range

        Returns:
            dict: Solution data
        """
        if equation_type == "ODE":
            x_test = np.linspace(domain[0], domain[1], 200)
            y_pred = self.model(x_test[:, np.newaxis]).numpy()
            return {
                'x': x_test.tolist(),
                'y': y_pred.flatten().tolist()
            }
        else:
            x = np.linspace(domain[0][0], domain[0][1], 50)
            t = np.linspace(domain[1][0], domain[1][1], 50)
            X, T = np.meshgrid(x, t)
            inputs = np.stack([X.flatten(), T.flatten()], axis=1)
            y_pred = self.model(inputs).numpy()
            return {
                'x': x.tolist(),
                't': t.tolist(),
                'y': y_pred.reshape(50, 50).tolist()
            }
