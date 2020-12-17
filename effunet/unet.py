import torch
import torch.nn as nn
import torchvision.transforms as T

def double_conv(in_,out_):
	conv = nn.Sequential(
		nn.Conv2d(in_,out_,kernel_size=3),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_,out_,kernel_size=3),
		nn.ReLU(inplace=True)
		)
	return conv

def crop(tensor,target_tensor):
	target_shape = target_tensor.shape[2]
	return T.CenterCrop(target_shape)(tensor)


class UNet(nn.Module):
	def __init__(self):
		super(UNet,self).__init__()

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.enc_conv_1 = double_conv(1,64)
		self.enc_convdouble_ = double_conv(64,128)
		self.enc_conv_3 = double_conv(128,256)
		self.enc_conv_4 = double_conv(256,512)
		self.enc_conv_5 = double_conv(512,1024)

		self.up_trans_1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
		self.dec_conv_1 = double_conv(1024,512)
		self.up_transdouble_ = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
		self.dec_convdouble_ = double_conv(512,256)
		self.up_trans_3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
		self.dec_conv_3 = double_conv(256,128)
		self.up_trans_4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
		self.dec_conv_4 = double_conv(128,64)

		self.out = nn.Conv2d(64,2,kernel_size=1)

	def forward(self, image):
		# Encoder
		x1 = self.enc_conv_1(image)
		x = self.pool(x1)
		x2 = self.enc_convdouble_(x)
		x = self.pool(x2)
		x3 = self.enc_conv_3(x)
		x = self.pool(x3)
		x4 = self.enc_conv_4(x)
		x = self.pool(x4)
		x = self.enc_conv_5(x)

		#Decoder
		x = self.up_trans_1(x)
		x = self.dec_conv_1(torch.cat([x,crop(x4,x)],axis=1))
		x = self.up_transdouble_(x)
		x = self.dec_convdouble_(torch.cat([x,crop(x3,x)],axis=1))
		x = self.up_trans_3(x)
		x = self.dec_conv_3(torch.cat([x,crop(x2,x)],axis=1))
		x = self.up_trans_4(x)
		x = self.dec_conv_4(torch.cat([x,crop(x1,x)],axis=1))
		
		#out
		x = self.out(x)
		return x


# image = torch.rand((1,1,572,572))
# model = UNet()
# out = model(image)
# print(out.shape)




		 
