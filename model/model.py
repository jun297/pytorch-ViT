import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class PatchEmbedding(nn.Module):
    """
    Split image into patches and then embed them through linear projection.

    Parameters
    ----------
    img_size : int 
        size of the image, it is a square
    
    patch_size : int
        size of the patch
    
    in_chans : int
        number of input channels

    embed_dim : int
        size of embedding dimension

    Attributes
    ----------
    n_patches : int
        number of patches divided from the input image
    
    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches and their embedding
    """
    def __init__(self, img_size, patch_size, in_chans = 3, embed_dim = 768):
        super.__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size = patch_size,
            stride = patch_size
        )
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            shape (n_samples, in_chnas, img_size ,img_size)
            just a batch of images
        
        Returns
        -------
        torch.Tensor
            shape (n_samples, n_patches, embed_dim)
        """

        x = self.proj(x) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(dim=2) # (n_samples, embed_dim, n_patches)
        x = x.transpose(1,2) # (n_samples, n_patches, embed_dim)

        return x


class Attention(nn.Module):
    """
    """




class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
