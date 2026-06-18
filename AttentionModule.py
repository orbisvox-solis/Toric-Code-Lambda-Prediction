# implementation of attention module used for determining lambda values
import torch
import torch.nn as nn
from math import sqrt

class AttentionModule(
    nn.Module
):
    
    # overwrite init method
    def __init__(
        self,
        lattice_size : int
    ):
        
        super( AttentionModule , self).__init__()
        self.lattice_size = lattice_size
        self.dim = 2 * self.lattice_size ** 2
    # forward method of attention module
    # shape of key and query: (batch_size , self.dim )
    # shape of value is : (batch_size, self.dim , self.dim)

    def forward(
        self,
        key : torch.Tensor,
        query : torch.Tensor,
        value : torch.Tensor
    ):
        
        # find attention weight and normalize it
        attention_weight = torch.einsum(
            "bx,by->bxy",
            key,
            query
        )/sqrt( self.dim )
        
        attention_weight = torch.softmax(
            attention_weight,
            dim = -1
        )

        # compute attention score
        attention_score = torch.einsum(
            "bxy,bzy->bxz",
            attention_weight,
            value
        )

        return attention_score
    
# check code correnctness
if __name__ == "__main__":

    batch_size = 100
    lattice_size = 4

    device = torch.device( "cuda" )
    attention_module = AttentionModule(
        lattice_size = lattice_size
    ).to( device = device )


    dummy_key = torch.randn(
        size = (batch_size , 2 * lattice_size ** 2)
    ).to( device = device )
    dummy_query = torch.randn(
        size = (batch_size , 2 * lattice_size ** 2)
    ).to( device = device )
    dummy_value = torch.randn(
        size = (batch_size , 2 * lattice_size ** 2, 2 * lattice_size ** 2)
    ).to( device = device )

    res = attention_module.forward(
        key = dummy_key,
        query = dummy_query,
        value = dummy_value
    )

    print( res )
    print( res.shape )
