# predictor of lambda set by having Ground basies and expecation value
# import required modules
import torch
import torch.nn as nn
from KeyGenerator import KeyGenerator
from QueryGenerator import QueryGenerator
from AttentionModule import AttentionModule
from ConvolutionalBlock import ConvolutionalBlock


# class Predictor
class Predictor(
    nn.Module
):
    
    # over-write init method
    def __init__(
        self,
        lattice_size : int
    ):
        
        # call base class init
        super( Predictor , self).__init__()

        # set important variables
        self.lattice_size = lattice_size
        self.attention_dim = 2 * self.lattice_size ** 2 


        # key and query generator
        self.key_generator = KeyGenerator(
            lattice_size = self.lattice_size,
            attention_dim = self.attention_dim
        )

        self.query_generator = QueryGenerator(
            lattice_size = self.lattice_size,
            attention_dim = self.attention_dim
        )

        # self attention module computer
        self.attention_computer = AttentionModule(
            lattice_size = self.lattice_size
        )

        # 4 block of Convolutional Block
        # the shapes are written for lattice_size = 8
        # shape : ( b , 128 , 4 , 8 , 8)
        self.convolutional_block_1 = ConvolutionalBlock(
            in_channel = self.attention_dim,
            channel_out = self.attention_dim,
            conv_kernel_size = (3 , 3 , 3),
            conv_padding_size = ( 2 , 1 , 1),
            pooling_kernel_size = (1 , 3 , 3),
            pooling_padding_size = ( 0 , 1 , 1),
            pooling_stride_size = ( 1 , 1 , 1),
            pool_type = "Avg"
        )
        
        # shape : ( b ,128 , 4 , 8 , 8)
        self.convolutional_block_2 = ConvolutionalBlock(
            in_channel = self.attention_dim,
            channel_out = self.attention_dim,
            conv_kernel_size = (3 , 3 , 3),
            conv_padding_size = ( 1 , 1 , 1),
            pooling_kernel_size = (1 , 3 , 3),
            pooling_padding_size = ( 0 , 1 , 1),
            pooling_stride_size = ( 1 , 1 , 1),
            pool_type = "Avg"
        )

        # shape : ( b , 128 , 4 , 4 , 4)
        self.convolutional_block_3 = ConvolutionalBlock(
            in_channel = self.attention_dim,
            channel_out = self.attention_dim,
            conv_kernel_size = (3 , 3 , 3),
            conv_padding_size = ( 1 , 1 , 1),
            pooling_kernel_size = (1 , 2 , 2),
            pooling_padding_size = ( 0 , 0 , 0),
            pooling_stride_size = ( 1 , 2 , 2),
            pool_type = "Max"
        )
        
        # shape : ( b , 48 * 128 , 4 , 2 , 2)
        self.convolutional_block_4 = ConvolutionalBlock(
            in_channel = self.attention_dim,
            channel_out = self.attention_dim,
            conv_kernel_size = (3 , 3 , 3),
            conv_padding_size = ( 1 , 1 , 1),
            pooling_kernel_size = (1 , 3 , 3),
            pooling_padding_size = ( 0 , 1 , 1),
            pooling_stride_size = ( 1 , 1 , 1),
            pool_type = "Avg"
        )

        # shape : ( b ,128 , 4 , 4 , 4)
        self.convolutional_block_5 = ConvolutionalBlock(
            in_channel = self.attention_dim,
            channel_out = self.attention_dim,
            conv_kernel_size = (3 , 3 , 3),
            conv_padding_size = ( 1 , 1 , 1),
            pooling_kernel_size = (1 , 3 , 3),
            pooling_padding_size = ( 0 , 1 , 1),
            pooling_stride_size = ( 1 , 1 , 1),
            pool_type = "Avg"
        )

        # shape : ( b ,  128 , 4 , 2 , 2)
        self.convolutional_block_6 = ConvolutionalBlock(
            in_channel = self.attention_dim,
            channel_out = self.attention_dim,
            conv_kernel_size = (3 , 3 , 3),
            conv_padding_size = ( 1 , 1 , 1),
            pooling_kernel_size = (1 , 2 , 2),
            pooling_padding_size = ( 0 , 0 , 0),
            pooling_stride_size = ( 1 , 2 , 2),
            pool_type = "Max"
        )

        # shape : ( b , 128 , 4 , 2 , 2)
        self.convolutional_block_7 = ConvolutionalBlock(
            in_channel = self.attention_dim,
            channel_out = self.attention_dim,
            conv_kernel_size = (3 , 3 , 3),
            conv_padding_size = ( 1 , 1 , 1),
            pooling_kernel_size = (1 , 2 , 2),
            pooling_padding_size = ( 0 , 1 , 1),
            pooling_stride_size = ( 1 , 1 , 1),
            pool_type = "Avg"
        )

        # shape : ( b , 128 , 4 , 3 , 3)
        self.convolutional_block_8 = ConvolutionalBlock(
            in_channel = self.attention_dim,
            channel_out = self.attention_dim,
            conv_kernel_size = (3 , 3 , 3),
            conv_padding_size = ( 1 , 1 , 1),
            pooling_kernel_size = (1 , 3 , 3),
            pooling_padding_size = ( 0 , 1 , 1),
            pooling_stride_size = ( 1 , 1 , 1),
            pool_type = "Avg"
        )

        # shape : ( b , 128 , 2 , 1 , 1)
        self.convolutional_block_9 = ConvolutionalBlock(
            in_channel =  self.attention_dim,
            channel_out = self.attention_dim,
            conv_kernel_size = (3 , 3 , 3),
            conv_padding_size = ( 1 , 1 , 1),
            pooling_kernel_size = (2, 3 , 3),
            pooling_padding_size = ( 0 , 0 , 0),
            pooling_stride_size = ( 2 , 3 , 3),
            pool_type = "Max"
        )

        self.flatten_layer = nn.Flatten()

        self.linear_layer_1 = nn.Linear(
            in_features = 256,
            out_features = 200
        )

        self.linear_layer_2 = nn.Linear(
            in_features = 200 , 
            out_features = 128
        )

        self.linear_layer_3 = nn.Linear(
            in_features = 128,
            out_features = 64
        )

        self.linear_layer_4 = nn.Linear(
            in_features = 64,
            out_features = 32 
        )

    # forward method of Predictor
    # shape of ground basis is (batch_size , 2L^2 , 2L^2) 
    # shape of amplitude is (batch_size , 2L^2)
    def forward(
        self,
        ground_basis : torch.Tensor,
        amplitude : torch.Tensor
    ) -> torch.Tensor :
        
        # convert ground basis into better format
        gs = 1 - 2 * ground_basis


        # generate key and query from amplitude
        key = self.key_generator.forward(
            x = amplitude
        )

        query = self.query_generator.forward(
            x = amplitude
        )

        # compute attentional input
        attention_features = self.attention_computer.forward(
            key = key,
            query = query,
            value = gs
        )

        
        attention_features = attention_features.reshape( (attention_features.shape[0] , attention_features.shape[1] , 2 , self.lattice_size , self.lattice_size))

        # pass throught 9 convolutional block
        features = self.convolutional_block_1.forward(
            x = attention_features
        )
        features = self.convolutional_block_2.forward(
            x = features
        )
        features = self.convolutional_block_3.forward(
            x = features
        )
        features = self.convolutional_block_4.forward(
            x = features
        )
        features = self.convolutional_block_5.forward(
            x = features
        )
        features = self.convolutional_block_6.forward(
            x = features
        )
        features = self.convolutional_block_7.forward(
            x = features
        )
        features = self.convolutional_block_8.forward(
            x = features
        )
        features = self.convolutional_block_9.forward(
            x = features
        )

        features = self.flatten_layer.forward(
            input = features
        )

        # pass through mlp 
        features = self.linear_layer_1( features )
        features = torch.tanh( features ) + features 
        
        features = self.linear_layer_2( features )
        features = torch.relu( features ) + features 

        features = self.linear_layer_3( features )
        features = torch.relu( features ) + features  

        result = self.linear_layer_4( features )
        result = torch.sigmoid( result ) + result

        return result
    

if __name__ == "__main__":

    device = torch.device( "cuda" )

    batch_size = 1
    lattice_size = 8

    dummy_gs = torch.randn( (batch_size , 2 * lattice_size ** 2 , 2 * lattice_size ** 2) ).to( device = device)
    amps = torch.randn( (batch_size , 2 * lattice_size ** 2) ).to( device = device)

    predictor = Predictor(
        lattice_size = lattice_size
    ).to( device = device)

    res = predictor.forward(
        ground_basis = dummy_gs,
        amplitude = amps
    )

    print( res )