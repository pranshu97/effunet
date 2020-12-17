import torch
import torch.nn as nn
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet


# Utility Functions for the model
def double_conv(in_,out_): # Double convolution layer for decoder 
	conv = nn.Sequential(
		nn.Conv2d(in_,out_,kernel_size=3),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_,out_,kernel_size=3),
		nn.ReLU(inplace=True)
		)
	return conv

def crop(tensor,target_tensor): # Crop tensor to target tensor size
	target_shape = target_tensor.shape[2]
	return T.CenterCrop(target_shape)(tensor)


# Hook functions to get values of intermediate layers for cross connection
hook_values = []
def hook(_, input, output):
	global hook_values
	hook_values.append(output) # stores values of each layers in hook_values

indices = []
shapes = []
def init_hook(model,device):
	global shapes, indices, hook_values

	for i in range(len(model._blocks)):
		model._blocks[i].register_forward_hook(hook) #register hooks
	
	image = torch.rand([1,3,572,572])
	image = image.to(device)
	out = model(image) # generate hook values to get shapes
	
	shape = [i.shape for i in hook_values] # get shape of all layers
	
	for i in range(len(shape)-1):
		if shape[i][2]!=shape[i+1][2]: # get indices of layers only where output dimension change
			indices.append(i)
	indices.append(len(shape)-1) # get last layer index
	
	shapes = [shape[i] for i in indices] # get shapes of required layers
	shapes = shapes[::-1]  

encoder_out = []
def epoch_hook(model, image):
	global encoder_out, indices, hook_values
	hook_values = []

	out = model(image) # generate layer outputs with current image
	encoder_out = [hook_values[i] for i in indices] # get layer outputs for selected indices


class EffUNet(nn.Module):

    def __init__(self,model='b0',out_channels=2,freeze_backbone=True,device='cuda'):
        super(EffUNet,self).__init__()
        global layers, shapes

        if model not in set(['b0','b1','b2','b3','b4','b5','b6','b7']):
            raise Exception(f'{model} unavailable.')
        self.encoder = EfficientNet.from_pretrained(f'efficientnet-{model}')

        # Disable non required layers by replacing them with identity
        self.encoder._conv_head=torch.nn.Identity()
        self.encoder._bn1=torch.nn.Identity()
        self.encoder._avg_pooling=torch.nn.Identity()
        self.encoder._dropout=torch.nn.Identity()
        self.encoder._fc=torch.nn.Identity()
        self.encoder._swish=torch.nn.Identity()

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.encoder.to(self.device)
        self.encoder._conv_stem.stride=1 # change stride of first layer from 2 to 1 to increase o/p size

		# freeze encoder
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # register hooks & get shapes
        init_hook(self.encoder,self.device)

        # Building decoder
        self.decoder = torch.nn.modules.container.ModuleList()
        for i in range(len(shapes)-1):
            self.decoder.append(torch.nn.modules.container.ModuleList())
            self.decoder[i].append(nn.ConvTranspose2d(shapes[i][1],shapes[i][1]-shapes[i+1][1],kernel_size=2,stride=2).to(self.device))
            self.decoder[i].append(double_conv(shapes[i][1],shapes[i+1][1]).to(self.device))

		#output layer
        self.out = nn.Conv2d(shapes[-1][1],out_channels,kernel_size=1).to(self.device)

    def forward(self, image):
        global layers
        shape = image.shape
        # Encoder
        epoch_hook(self.encoder, image) # required outputs accumulate in "encoder_out"

        #Decoder
        x = encoder_out.pop()
        for i in range(len(self.decoder)):
            x = self.decoder[i][0](x) # conv transpose
            prev = encoder_out.pop()
            prev = crop(prev,x) # croping for cross connection
            prev = torch.cat([x,prev],axis=1) # concatenating 
            x = self.decoder[i][1](prev) # double conv
		
		#out
        x = self.out(x)
        return x

# img = torch.rand([1,3,572,572]).cuda()
# model = EffUNet()
# print(model)
# out = model(img)
# print(out.shape)