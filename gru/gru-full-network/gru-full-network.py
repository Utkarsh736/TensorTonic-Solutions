import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)
        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> dict:
        """
        Returns outputs and final_hidden_state as float64 arrays.
        """
        X = np.asarray(X, dtype=np.float64)

        N, T, _ = X.shape
        H = self.hidden_dim

        h = np.zeros((N, H), dtype=np.float64)

        step_outputs = []

        for t in range(T):
            x_t = X[:, t, :] 
            joined_gates = np.concatenate([h, x_t], axis=-1)    

            r_t = sigmoid(joined_gates @ self.W_r.T + self.b_r) 

            z_t = sigmoid(joined_gates @ self.W_z.T + self.b_z) 

            gated_h = r_t * h                                   
            joined_candidate = np.concatenate([gated_h, x_t], axis=-1)
            h_tilde = np.tanh(joined_candidate @ self.W_h.T + self.b_h)  

            h = z_t * h + (1.0 - z_t) * h_tilde            

            y_t = h @ self.W_y.T + self.b_y                    
            step_outputs.append(y_t)

        outputs = np.stack(step_outputs, axis=1)               

        return {
            "outputs": np.asarray(outputs, dtype=np.float64),
            "final_hidden_state": np.asarray(h, dtype=np.float64),
        }