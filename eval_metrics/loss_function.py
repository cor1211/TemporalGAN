import torch

import torch
import torch.nn as nn

def reverse_map(change_map):
    """
    Reverses the change map so that the changed pixels have a weight of Min and the unchanged pixels have a weight of Max.
    
    Args:
    - change_map: A PyTorch tensor of size (batch_size, c, height, width) representing the change weight map.
    
    Returns:
    - reversed_change_map: A PyTorch tensor of size (batch_size, c, height, width) representing the reversed change weight map.
    """
    # Find the maximum and minimum for each channel in the change map | although S1 is one channel but this can come in handy if we want to VH polarization in the future.
    max_values, _ = torch.max(change_map, dim=3, keepdim=True)
    max_values, _ = torch.max(max_values, dim=2, keepdim=True)
    min_values, _ = torch.max(change_map, dim=3, keepdim=True)
    min_values, _ = torch.max(min_values, dim=2, keepdim=True)
    reversed_change_map = max_values - change_map + min_values
    return reversed_change_map

class WeightedL1Loss(nn.Module):
    def __init__(self, change_weight = 5, convert_to_float32: bool = True, legacy_chage_map: bool = False):
        """
        Args
        ----
        change_weight: A scalar value representing the weight of L1 loss for changed pixels.
                        the weight of L1 loss for unchanged pixels is 1.
        convert_to_float32: A boolean value representing whether to convert the input and target images to float32.
                            This is useful for when the input and target images are float16 which can cause the loss to be NaN.
        legacy_chage_map: A boolean value representing whether to use the legacy change map or not.
                            The legacy change map is calculated as (1-change_map) instead of (max(change_map) - change_map + min(change_map))
            
                        

        Returns:
        - None
        """
        super().__init__()
        self.change_weight = change_weight
        self.convert_to_float32 = convert_to_float32
        self.legacy_chage_map = legacy_chage_map
    
        
    def forward(self, input, target, change_map):
        """
        Calculates the L1 loss between the input and target images using a change map.

        Args:
        - input: A PyTorch tensor of size (batch_size, channels, height, width) representing the input image.
        - target: A PyTorch tensor of size (batch_size, channels, height, width) representing the target image.
        - change_map: A PyTorch tensor of size (batch_size, 1, height, width) representing the change weight map.
        
        Attributes:
        - reversed_change_map: A PyTorch tensor of size (batch_size, C, height, width) representing the reversed change weight map.


        Returns:
        - loss: A PyTorch scalar representing the weighted L1 loss.
        """
        if self.legacy_chage_map:
            reversed_change_map = (1-change_map.clone())
        else:
            cm_copy = change_map.clone()
            # Find the maximum and minimum for each channel in the change map | although S1 is one channel but this can come in handy if we want to VH polarization in the future.
            reversed_change_map = reverse_map(cm_copy)
            
        if self.convert_to_float32:
            input = input.to(torch.float64)
            target = target.to(torch.float64)
            change_map = change_map.to(torch.float64)
            reversed_change_map = reversed_change_map.to(torch.float64)
            
        # Calculate the absolute difference between the input and target images
        abs_diff = torch.abs(input - target)
        
        # Calculate the mean of the change map along the channels dimension so the weights are a 2D tensor (batch, 1, height, width)
        change_map = torch.mean(change_map, dim=1, keepdim=True)
        reversed_change_map = torch.mean(reversed_change_map, dim=1, keepdim=True)
        
        # Multiply the absolute difference by the cahnge map
        change_weighted_diff = abs_diff * change_map
        # Sum the weighted differences along the height and width dimensions
        sum_change_weighted_diff = torch.sum(change_weighted_diff, dim=[2, 3])
        # Sum the weights along the height and width dimensions
        sum_weights = torch.sum(change_map, dim=[2, 3])
        sum_weights.masked_fill_(sum_weights == 0, 0.0001)
        # Divide the sum of the weighted differences by the sum of the weights
        changed_loss = torch.mean(sum_change_weighted_diff / sum_weights)
        
        
        # Multiply the absolute difference by the unhanged map
        unchange_weighted_diff = abs_diff * reversed_change_map
        # Sum the weighted differences along the height and width dimensions
        sum_unchange_weighted_diff = torch.sum(unchange_weighted_diff, dim=[2, 3])
        # Sum the weights along the height and width dimensions
        sum_weights = None
        sum_weights = torch.sum(reversed_change_map, dim=[2, 3])
        sum_weights.masked_fill_(sum_weights == 0, 0.0001)
        # Divide the sum of the weighted differences by the sum of the weights
        unchanged_loss = torch.mean(sum_unchange_weighted_diff / sum_weights)
    
        # Calculate the final loss as a weighted sum of the changed and unchanged losses
        loss = (unchanged_loss + self.change_weight * changed_loss) / (1 + self.change_weight)
        
        if torch.isnan(loss) or torch.isnan(loss).any():
            raise ValueError(f"Loss is NaN \n \
                            changed_loss: {torch.mean(changed_loss)} | unchanged_loss: {torch.mean(unchanged_loss)} \n \
                            sum_unchange_weighted_diff: {torch.mean(sum_unchange_weighted_diff)} | sum_weights: {torch.mean(sum_weights)} \n \
                            abs_diff: {torch.mean(abs_diff)} | change_map: {torch.mean(change_map)} | reversed_change_map: {torch.mean(reversed_change_map)} \n \
                            input: {torch.mean(input)} | target: {torch.mean(target)}")
        
        return loss.to(torch.float64)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 variant).
    Formula: sqrt((x - y)^2 + epsilon^2)
    This is more robust to outliers and prevents dead zones around 0, leading to
    smoother convergence than standard L1 loss.
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG19. 
    It measures the difference between feature maps of generated and ground-truth images.
    """
    def __init__(self, feature_layers=['relu2_2', 'relu3_3', 'relu4_3'], use_cuda=True):
        super(PerceptualLoss, self).__init__()
        
        # Load VGG19 pretrained on ImageNet
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        # Define the splits based on layer names
        # VGG19 maxpool indices: 4, 9, 18, 27, 36
        # 'relu1_1': 1, 'relu1_2': 3
        # 'relu2_1': 6, 'relu2_2': 8
        # 'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17
        # 'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26
        # 'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35
        
        layer_indices = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35
        }
        
        # Build slices
        for x in range(layer_indices.get('relu1_2', 4)):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(layer_indices.get('relu1_2', 4), layer_indices.get('relu2_2', 9)):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(layer_indices.get('relu2_2', 9), layer_indices.get('relu3_3', 16)):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(layer_indices.get('relu3_3', 16), layer_indices.get('relu4_3', 25)):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(layer_indices.get('relu4_3', 25), layer_indices.get('relu5_3', 34)):
            self.slice5.add_module(str(x), vgg[x])
            
        if use_cuda:
            self.slice1.cuda()
            self.slice2.cuda()
            self.slice3.cuda()
            self.slice4.cuda()
            self.slice5.cuda()
            
        for param in self.parameters():
            param.requires_grad = False
            
        self.feature_layers = feature_layers
        self.criterion = nn.L1Loss()
        
        # ImageNet normalization parameters (required for VGG)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        # Assumes x is in range [-1, 1] as standard for GANs in this pipeline
        # Map [-1, 1] to [0, 1] first
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, X, Y):
        # If inputs are 1 channel (SAR), repeat to 3 channels for VGG
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        if Y.shape[1] == 1:
            Y = Y.repeat(1, 3, 1, 1)
            
        X = self._normalize(X)
        Y = self._normalize(Y)
        
        h_x = self.slice1(X)
        h_y = self.slice1(Y)
        out_x_1 = h_x
        out_y_1 = h_y
        
        h_x = self.slice2(h_x)
        h_y = self.slice2(h_y)
        out_x_2 = h_x
        out_y_2 = h_y
        
        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        out_x_3 = h_x
        out_y_3 = h_y
        
        h_x = self.slice4(h_x)
        h_y = self.slice4(h_y)
        out_x_4 = h_x
        out_y_4 = h_y
        
        h_x = self.slice5(h_x)
        h_y = self.slice5(h_y)
        out_x_5 = h_x
        out_y_5 = h_y
        
        features_x = {'relu1_2': out_x_1, 'relu2_2': out_x_2, 'relu3_3': out_x_3, 'relu4_3': out_x_4, 'relu5_3': out_x_5}
        features_y = {'relu1_2': out_y_1, 'relu2_2': out_y_2, 'relu3_3': out_y_3, 'relu4_3': out_y_4, 'relu5_3': out_y_5}
        
        loss = 0.0
        for layer in self.feature_layers:
            if layer in features_x:
                loss += self.criterion(features_x[layer], features_y[layer])
                
        return loss
if __name__ == "__main__":
    # Create a dummy input and target image
    input = torch.rand((1, 3, 256, 256)).to(torch.float16)
    target = torch.rand((1, 3, 256, 256)).to(torch.float16)

    # Create a dummy weight map
    change_map = torch.rand((1, 3, 256, 256)).to(torch.float16)
    print("mean->", torch.min(change_map), change_map.shape, change_map.dtype)
    # Calculate the weighted L1 loss
    loss_dytpe16 = WeightedL1Loss(change_weight=1,convert_to_float32=False)(input, target, change_map)
    loss_dytpe32 = WeightedL1Loss(change_weight=1,convert_to_float32=True)(input, target, change_map)

    print("Loss with float16: ", loss_dytpe16)
    print("Loss with float32: ", loss_dytpe32)