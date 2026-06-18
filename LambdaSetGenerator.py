import numpy as np


# generating lambda values in shape of (size , 2 , lattice_size ^ 2 , lattice_size ^ 2)
def generate_lambda_set(
    lattice_size : int,
    size : int
):
    

    lambda_set = np.random.uniform(
        low = 0,
        high = 1,
        size = ( size , 2 * lattice_size  ** 2)
    )

    return lambda_set

def generate_masked_lambda_set(
    lattice_size : int,
    size : int,
    masked_value : float      
):
    
    lambda_set = generate_lambda_set(
        lattice_size = lattice_size,
        size = size
    )

    mask_set = generate_lambda_set(
        lattice_size = lattice_size,
        size = size
    )

    masked_set = np.where(
        mask_set > masked_value,
        0,
        lambda_set
    )

    masked_set = np.round( masked_set , decimals = 5)
    return masked_set


if __name__ == "__main__":

    lambda_set = generate_masked_lambda_set(
        lattice_size = 8,
        size = 1000,
        masked_value = 0.32
    )

    print( lambda_set.shape )
    

