# convolutoinal block implementation
# each block has 1 conv3d layer, with mish activation, 1 average/max pooling layer and 
import torch
import torch.nn as nn


class ConvolutionalBlock(
    nn.Module
):
    
    # overwrite init
    def __init__(
        self,
        in_channel : int,
        channel_out : int,
        conv_kernel_size : tuple[int],
        conv_padding_size : tuple[int],
        pooling_kernel_size : tuple[int],
        pooling_stride_size : tuple[int],
        pooling_padding_size : tuple[int],
        pool_type : str = "Avg"
    ):
        
        # call base class init method
        super( ConvolutionalBlock, self).__init__()
        
        # define mish function
        self.mish_function = nn.Mish()

        # define layers
        self.conv3d_layer = nn.Conv3d(
            in_channels = in_channel,
            out_channels = channel_out,
            kernel_size = conv_kernel_size,
            padding = conv_padding_size,
        )

        self.batch_normalization_layer = nn.BatchNorm3d(
            num_features = in_channel
        )
        
        if pool_type == "Avg":

            self.pooling_layer = nn.AvgPool3d(
                kernel_size = pooling_kernel_size,
                stride = pooling_stride_size,
                padding = pooling_padding_size
            )

        elif pool_type == "Max":

            self.pooling_layer = nn.MaxPool3d(
                kernel_size = pooling_kernel_size,
                stride = pooling_stride_size,
                padding = pooling_padding_size
            )

        else:
            raise ValueError(
                "Unexpected pooling layer type"
            )
        
    # forward method of ConvoltionalBlock
    def forward(
        self,
        x : torch.Tensor
    ) -> torch.Tensor :
        
        # normalized input data
        normalized_input = self.batch_normalization_layer.forward(
            input = x
        )

        # apply conv3d
        res = self.conv3d_layer.forward(
            input = normalized_input
        )


        # apply mish + residual connection
        res = self.mish_function.forward(
            input = res
        ) + res


        # apply pooling layer
        res = self.pooling_layer.forward(
            input = res
        )

        return res
    


if __name__ == "__main__":


    dummy_data = torch.randn(
        size = ( 1 , 32 , 2 , 4 , 4)
    )

    cb = ConvolutionalBlock(
        in_channel = 32,
        channel_out = 128,
        conv_kernel_size = ( 5 , 3 , 3),
        conv_padding_size = ( 2 , 1 , 1),
        pooling_kernel_size = (1,3,3),
        pooling_padding_size = (0 ,1,1),
        pooling_stride_size = (1,1,1)
    )


    res = cb.forward(
        x = dummy_data
    )

    print( res )
    print( res.shape )