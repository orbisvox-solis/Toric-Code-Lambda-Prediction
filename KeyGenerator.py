# generating key for self attention mechanism
# import modules
import torch
import torch.nn as nn
from LinearProjector import LinearProjector


# class for generating key

class KeyGenerator(
    nn.Module
):
    

    # overwrite init method
    def __init__(
        self,
        lattice_size : int,
        attention_dim : int
    ):
        
        # call base init method
        super( KeyGenerator, self).__init__()

        self.lattice_size = lattice_size
        self.attention_dim = attention_dim
        self.input_dim = 2 * self.lattice_size ** 2

        self.linear_projector = LinearProjector(
            in_features = self.input_dim,
            out_features = self.attention_dim
        )


    # apply forward method for generating key from coeffienet
    # input shape is (batch_size , input_dim)
    # output shape is (batch_size , attention_dim)
    def forward(
        self,
        x : torch.Tensor
    ) -> torch.Tensor:
        
        res = self.linear_projector.forward(
            x = x
        )

        # apply tanh followed by residual connection
        return torch.tanh( res ) + res 
    

if __name__ == "__main__":

    # initlize device 
    device = torch.device("cuda")

    batch_size = 100
    lattice_size = 4
    attention_dim = 20

    key_generator = KeyGenerator(
        lattice_size = lattice_size,
        attention_dim = attention_dim
    ).to( device = device )

    dummy_data = torch.randn(
        size = (batch_size , 2 * lattice_size ** 2)
    ).to( device = device )

    key = key_generator.forward(
        x = dummy_data
    )

    print( key )
    print( key.shape )