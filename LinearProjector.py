##### implementation of linear projector with pytorch
# import required modules
import torch.nn as nn
import torch

class LinearProjector(
    nn.Module
):
    

    # define init method
    def __init__(
        self,
        in_features : int,
        out_features : int
    ):
        

        # call base class instructor
        super( LinearProjector, self).__init__()

        self.linear_layer = nn.Linear(
            in_features = in_features,
            out_features = out_features
        )

    
    # define forward method
    # x is input tensor with size of (batch_size, in_features)
    # output of forward method is (batch_size, out_features) 
    def forward(
        self,
        x : torch.Tensor
    ) -> torch.Tensor :
        
        res = self.linear_layer.forward(
            input = x
        )

        return res
    


# test for checking correctness of code
if __name__ == "__main__":


    batch_size = 100
    in_features = 20 
    out_features = 15

    device = torch.device( "cuda" )

    lp = LinearProjector(
        in_features = in_features,
        out_features = out_features
    ).to( device = device )

    dummy_input = torch.randn(
        size = (batch_size, in_features)
    ).to( device = device )

    res = lp.forward(
        x = dummy_input
    )

    print( res )
    print( res.shape )