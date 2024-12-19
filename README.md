# PINN Equation Solver API

This API provides a service for solving Ordinary Differential Equations (ODEs) and Partial Differential Equations (PDEs) using Physics-Informed Neural Networks (PINNs).

---

## **How to Run the Project**

### **Requirements**

Flask==2.0.1
Werkzeug==2.0.3
flask-cors==3.0.10
tensorflow>=2.8.0
numpy>=1.19.2

### **Setup**

1. Clone the repository:

   ```bash
   git clone https://github.com/DiaeEddineJamal/equation_solver.git
   
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python main.py
   ```

4. By default, the server runs on `http://localhost:5000`. You can customize the port using the `PORT` environment variable.

---

## **API Endpoints**

### **1. Health Check**

#### **URL**

`GET /health`

#### **Description**

Checks if the API is running and healthy.

#### **Response**

```json
{
    "status": "healthy",
    "service": "PINN Equation Solver",
    "version": "1.0.0"
}
```

---

### **2. Initialize Model**

#### **URL**

`POST /api/initialize`

#### **Description**

Initializes the PINN model with specified configurations.

#### **Request Body**

For ODE:

```json
{
    "layers": [10, 20, 20, 10, 1],
    "equation_type": "ODE"
}
```

For PDE:

```json
{
    "layers": [10, 20, 20, 10, 1],
    "equation_type": "PDE"
}
```

#### **Response**

```json
{
    "status": "success",
    "data": {
        "status": "success",
        "message": "Model initialized successfully",
        "details": {
            "equation_type": "ODE",
            "layers": [10, 20, 20, 10, 1]
        }
    }
}
```

---

### **3. Solve Equation**

#### **URL**

`POST /api/solve`

#### **Description**

Solves the specified differential equation using the initialized PINN model.

#### **Request Body**

For ODE:

```json
{
    "equation_type": "ODE",
    "domain": [0, 1],
    "num_points": 100,
    "epochs": 500,
    "save_result": true,
    "result_id": "example_ode"
}
```

For PDE:

```json
{
    "equation_type": "PDE",
    "domain": [[0, 1], [0, 1]],
    "num_points": 100,
    "epochs": 500,
    "save_result": true,
    "result_id": "example_pde"
}
```

#### **Response (ODE)**

```json
{
    "status": "success",
    "data": {
        "status": "success",
        "losses": [0.123, 0.045, 0.003],
        "solution": {
            "x": [0.0, 0.01, ..., 1.0],
            "y": [0.0, 0.01, ..., 1.0]
        }
    }
}
```

#### **Response (PDE)**

```json
{
    "status": "success",
    "data": {
        "status": "success",
        "losses": [0.567, 0.234, 0.123],
        "solution": {
            "x": [0.0, 0.02, ..., 1.0],
            "t": [0.0, 0.02, ..., 1.0],
            "y": [[0.0, 0.01, ...], [0.01, 0.02, ...], ...]
        }
    }
}
```

---

### **4. Unavailable Endpoint (404)**

#### **URL**

`GET /api/non-existent`

#### **Description**

Accessing an unavailable endpoint returns an error.

#### **Response**

```json
{
    "status": "error",
    "message": "Endpoint not found"
}
```

---

## **Testing the API with Postman**

1. Import the API endpoints into Postman using the above details.
2. Use the following URLs for testing:
   - Health Check: `GET http://localhost:5000/health`
   - Initialize Model: `POST http://localhost:5000/api/initialize`
   - Solve Equation: `POST http://localhost:5000/api/solve`
3. Provide the respective JSON body for ODE and PDE endpoints during testing.
4. Verify the expected responses.

---

## **License and Copyright**

This project is protected under copyright. Unauthorized use or duplication of any part of this repository is prohibited without explicit permission.

**Copyright © 2024 Diae aka Luziv**

