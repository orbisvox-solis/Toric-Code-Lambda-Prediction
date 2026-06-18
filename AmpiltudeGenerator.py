import numpy as np
import torch

def compute_amplitude(
    states : np.ndarray,
    lambda_set : np.ndarray,
    beta : float
) -> torch.Tensor :
    

    sign = 1 - 2 * states
    
    # change to torch
    sign = torch.from_numpy( sign )
    lambda_set = torch.from_numpy( lambda_set )

    # compute exponent
    exponent = (beta / 2) * torch.einsum(
        "bi,bji->bj",
        lambda_set,
        sign
    )
    
    return exponent

if __name__ == "__main__":


    # Two basis states, two qubits
    states = np.random.random(
        size = (1 , 4 , 4)
    ) 

    # Three λ‑sets
    lambda_sets = np.random.random(
        size = (1 , 4)
    )
    print( 1 - 2 * states )
    print( lambda_sets )
    amps = compute_amplitude(states, lambda_sets , beta = 2.0 )
    print(amps )    