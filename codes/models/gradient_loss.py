import torch
import torch.nn as nn
from kornia.filters import laplacian


class GradientLoss(nn.Module):
    def __init__(self, gradient_type, horizontal, vertical, loss_type, device):
        super(GradientLoss, self).__init__()
        self.gradient_type = gradient_type
        self.horizontal = horizontal
        self.vertical = vertical
        self.device = device
        if loss_type == 'l1':
            self.loss = torch.nn.L1Loss(reduction='sum').to(device)
        elif loss_type == 'l2':
            self.loss = torch.nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss_type))
        return

    # When horizontal = 3, vertical = 3:
    # +────+────+────+
    # | 00 | 10 | 20 |
    # +────+────+────+
    # | 01 | 11 | 21 |
    # +────+────+────+
    # | 02 | 12 | 22 |
    # +────+────+────+
    # allow_lost_pixels: 当像素点数量无法正好整除的时候，是否将剩下的像素补充到最后的小块中
    def crop_image_to_grid(self, image, horizontal, vertical, allow_lost_pixels=False):
        # Input image: [Channel * Height * Width]
        image_height = image.shape[1]
        image_width = image.shape[2]
        image_whc = image.reshape([image_height, image_width, image.shape[0]])
        horizontal_block_pixels = image_width // horizontal  # 切割后每个小块的宽
        vertical_block_pixels = image_height // vertical  # 切割后每个小块的高
        is_horizontal_average = horizontal_block_pixels * horizontal == image_width
        is_vertical_average = vertical_block_pixels * vertical == image_height
        cropped_images = {}
        for i in range(horizontal):
            for j in range(vertical):
                left = i * horizontal_block_pixels
                up = j * vertical_block_pixels
                if (not allow_lost_pixels) and (not is_horizontal_average) and i == horizontal - 1:
                    right = image_width
                else:
                    right = left + horizontal_block_pixels
                if (not allow_lost_pixels) and (not is_vertical_average) and j == vertical - 1:
                    bottom = image_height
                else:
                    bottom = up + vertical_block_pixels
                cropped_block = image_whc[up:bottom, left:right]
                cropped_images[str(i) + '|' + str(j)] = cropped_block.reshape(
                    [cropped_block.shape[2], cropped_block.shape[0], cropped_block.shape[1]])
        return cropped_images

    def forward(self, ground_truth_batch, segmented_print_batch):
        # Input image: [Batch * Channel * Height * Width]
        if self.gradient_type == 'laplace':
            ground_truth_gradient_batch = laplacian(ground_truth_batch, kernel_size=3)
            segmented_print_gradient_batch = laplacian(segmented_print_batch, kernel_size=3)
        # elif self.gradient_type == 'sobel_x':
        #     pass
        # elif self.gradient_type == 'sobel_y':
        #     pass
        else:
            raise NotImplementedError('Gradient type [{:s}] not recognized.'.format(self.radient_type))
        loss = torch.tensor(.0).to(self.device)
        for i in range(len(ground_truth_batch)):
            ground_truth_gradient = ground_truth_gradient_batch[i]
            segmented_print_gradient = segmented_print_gradient_batch[i]
            ground_truth_gradient_cropped = self.crop_image_to_grid(ground_truth_gradient, self.horizontal,
                                                                    self.vertical)
            segmented_print_gradient_cropped = self.crop_image_to_grid(segmented_print_gradient, self.horizontal,
                                                                       self.vertical)
            max_key = list(ground_truth_gradient_cropped.keys())[0]
            max_loss = self.loss(ground_truth_gradient_cropped[max_key], segmented_print_gradient_cropped[max_key])
            for key in ground_truth_gradient_cropped.keys():
                ground_truth_image_gradient_cropped_tensor = ground_truth_gradient_cropped[key]
                segmented_print_image_gradient_cropped_tensor = segmented_print_gradient_cropped[key]
                loss_value = self.loss(ground_truth_image_gradient_cropped_tensor,
                                       segmented_print_image_gradient_cropped_tensor)
                if loss_value > max_loss:
                    max_key = key
                    max_loss = loss_value
            loss += self.loss(ground_truth_gradient_cropped[max_key].to(self.device),
                              segmented_print_gradient_cropped[max_key].to(self.device)).to(self.device)
        return loss
