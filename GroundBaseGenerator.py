import numpy as np



def get_qubit_index(
    x : int,
    y : int,
    lattice_size : int
):
    
    n_vertex = lattice_size ** 2

    # get index of qubit for specific vertex index
    right = y * lattice_size + x
    left = y * lattice_size + ( (x - 1) % lattice_size )

    up = x * lattice_size + y + n_vertex
    down = x * lattice_size + ((y-1) % lattice_size) + n_vertex

    return right, left, up, down




def generate_computational_basis_smaple(
    sample_count : int,
    lattice_size : int
) -> np.ndarray :
    

    n_vertex  = lattice_size ** 2
    n_edge = 2 * n_vertex
    

    current_computation_base = np.zeros( shape = (n_edge,) , dtype = np.uint8 )
    samples = [current_computation_base.copy() ]
    sample_count -= 1
    while True:

        vertex_order = np.random.permutation( n_vertex )

        index = min( sample_count , n_vertex )

        for i in range( index ):
            
            y = vertex_order[i] // lattice_size
            x = vertex_order[i] % lattice_size

            right, left, up, down = get_qubit_index(
                x = x,
                y = y,
                lattice_size = lattice_size
            )

            current_computation_base[[right,left,up,down]] ^= 1

            samples.extend( [current_computation_base.copy()] )

        sample_count -= index
        if sample_count == 0 :
            break
    
    samples = np.array( samples )
    
    return samples

if __name__ == '__main__':

    samples = generate_computational_basis_smaple(
        sample_count = 10000 * 128 ,
        lattice_size = 8
    )

    print( samples.shape)