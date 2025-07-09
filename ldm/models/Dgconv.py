import torch
import torch.nn as nn
import torch.nn.functional as F


class IrregularDirectionalGradientConv(nn.Module):
    """不规则方向梯度卷积 (IDG) - 用于捕获不规则纹理"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(IrregularDirectionalGradientConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 可学习的卷积核，用于不规则梯度提取
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=self.padding, bias=False)
        
        # 初始化为随机梯度检测器
        nn.init.xavier_normal_(self.conv.weight)
        
    def forward(self, x):
        # 使用可学习的卷积核提取不规则方向梯度
        return self.conv(x)


class CenterSurroundingGradientConv(nn.Module):
    """中心-周围梯度卷积 (CSG) - 用于对比度感知"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CenterSurroundingGradientConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 创建中心-周围梯度核
        self.register_buffer('csg_kernel', self._create_csg_kernel())
        
    def _create_csg_kernel(self):
        """创建中心-周围梯度核"""
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size) * (-1.0)
        center = self.kernel_size // 2
        kernel[0, 0, center, center] = float(self.kernel_size * self.kernel_size - 1)
        kernel = kernel / (self.kernel_size * self.kernel_size)
        return kernel
        
    def forward(self, x):
        # 扩展kernel到所有通道
        kernel = self.csg_kernel.expand(self.out_channels, self.in_channels // self.out_channels, 
                                       self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernel, padding=self.padding, groups=min(self.in_channels, self.out_channels))


class HorizontalGradientConv(nn.Module):
    """水平梯度卷积 (HG)"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(HorizontalGradientConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 创建水平梯度核（类似Sobel算子）
        self.register_buffer('h_kernel', self._create_h_kernel())
        
    def _create_h_kernel(self):
        """创建水平梯度检测核"""
        kernel = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        if self.kernel_size == 3:
            kernel[0, 0] = torch.tensor([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]], dtype=torch.float32)
        else:
            # 对于其他尺寸，创建简单的水平梯度
            kernel[0, 0, :, 0] = -1
            kernel[0, 0, :, -1] = 1
        return kernel / (self.kernel_size * 2)
        
    def forward(self, x):
        kernel = self.h_kernel.expand(self.out_channels, self.in_channels // self.out_channels,
                                     self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernel, padding=self.padding, groups=min(self.in_channels, self.out_channels))


class VerticalGradientConv(nn.Module):
    """垂直梯度卷积 (VG)"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(VerticalGradientConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 创建垂直梯度核
        self.register_buffer('v_kernel', self._create_v_kernel())
        
    def _create_v_kernel(self):
        """创建垂直梯度检测核"""
        kernel = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        if self.kernel_size == 3:
            kernel[0, 0] = torch.tensor([[-1, -2, -1],
                                        [ 0,  0,  0],
                                        [ 1,  2,  1]], dtype=torch.float32)
        else:
            # 对于其他尺寸，创建简单的垂直梯度
            kernel[0, 0, 0, :] = -1
            kernel[0, 0, -1, :] = 1
        return kernel / (self.kernel_size * 2)
        
    def forward(self, x):
        kernel = self.v_kernel.expand(self.out_channels, self.in_channels // self.out_channels,
                                     self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernel, padding=self.padding, groups=min(self.in_channels, self.out_channels))


class CenterSurroundingAggregationConv(nn.Module):
    """中心-周围聚合卷积 (CSA) - 用于对比度增强"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CenterSurroundingAggregationConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 可学习的聚合权重
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             padding=self.padding, bias=False)
        
        # 初始化为平均池化
        nn.init.constant_(self.conv.weight, 1.0 / (kernel_size * kernel_size))
        
    def forward(self, x):
        return self.conv(x)


class DGConvModule(nn.Module):
    """自适应方向梯度卷积模块 (DGConv)"""
    def __init__(self, in_channels, out_channels=None, kernel_size=3, 
                 use_vanilla=True, reduction_ratio=4):
        super(DGConvModule, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_vanilla = use_vanilla
        
        # 各种方向梯度卷积
        self.idg = IrregularDirectionalGradientConv(in_channels, out_channels, kernel_size)
        self.csg = CenterSurroundingGradientConv(in_channels, out_channels, kernel_size)
        self.hg = HorizontalGradientConv(in_channels, out_channels, kernel_size)
        self.vg = VerticalGradientConv(in_channels, out_channels, kernel_size)
        self.csa = CenterSurroundingAggregationConv(in_channels, out_channels, kernel_size)
        
        # 是否包含普通卷积
        if use_vanilla:
            self.vanilla_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                        padding=kernel_size//2, bias=False)
        
        # 自适应融合机制 - 使用SE-like注意力
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        num_convs = 6 if use_vanilla else 5
        
        # 通道注意力用于自适应融合
        self.fc1 = nn.Linear(out_channels, out_channels // reduction_ratio)
        self.fc2 = nn.Linear(out_channels // reduction_ratio, out_channels * num_convs)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # 1x1卷积用于最终特征整合
        self.fusion_conv = nn.Conv2d(out_channels * num_convs, out_channels, 1, bias=True)
        
    def forward(self, x):
        # 获取各个方向的梯度输出
        outputs = []
        
        idg_out = self.idg(x)
        csg_out = self.csg(x)
        hg_out = self.hg(x)
        vg_out = self.vg(x)
        csa_out = self.csa(x)
        
        outputs = [idg_out, csg_out, hg_out, vg_out, csa_out]
        
        if self.use_vanilla:
            vanilla_out = self.vanilla_conv(x)
            outputs.append(vanilla_out)
        
        # 堆叠所有输出
        stacked = torch.stack(outputs, dim=1)  # [B, num_convs, C, H, W]
        
        # 计算自适应权重（基于全局池化特征）
        gap = self.global_pool(x)  # [B, C, 1, 1]
        gap = gap.view(gap.size(0), -1)  # [B, C]
        
        # 通过FC层生成权重
        weights = self.fc1(gap)
        weights = self.relu(weights)
        weights = self.fc2(weights)  # [B, C * num_convs]
        weights = self.sigmoid(weights)
        
        # 重塑权重
        B, num_convs, C, H, W = stacked.shape
        weights = weights.view(B, num_convs, C, 1, 1)  # [B, num_convs, C, 1, 1]
        
        # 应用权重
        weighted = stacked * weights  # [B, num_convs, C, H, W]
        
        # 融合所有特征
        weighted = weighted.view(B, -1, H, W)  # [B, num_convs * C, H, W]
        output = self.fusion_conv(weighted)  # [B, out_channels, H, W]
        
        return output


class OptimizedDGConvModule(nn.Module):
    """优化版本的DGConv - 显存效率更高"""
    def __init__(self, channels, kernel_size=3, use_vanilla=True):
        super(OptimizedDGConvModule, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_vanilla = use_vanilla
        
        # 预定义的梯度卷积核
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2], 
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0) / 8.0)
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0) / 8.0)
        
        self.register_buffer('laplacian', torch.tensor([
            [[ 0, -1,  0],
             [-1,  4, -1],
             [ 0, -1,  0]]
        ], dtype=torch.float32).unsqueeze(0) / 6.0)
        
        self.register_buffer('center_surround', torch.tensor([
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]]
        ], dtype=torch.float32).unsqueeze(0) / 9.0)
        
        # IDG使用可学习的卷积
        self.idg_conv = nn.Conv2d(channels, channels, kernel_size,
                                 padding=self.padding, groups=channels, bias=False)
        
        # 普通卷积（可选）
        if use_vanilla:
            self.vanilla_conv = nn.Conv2d(channels, channels, kernel_size,
                                        padding=self.padding, groups=channels, bias=False)
        
        # 自适应融合机制
        num_features = 6 if use_vanilla else 5
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * num_features, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Conv2d(channels, channels, 1, bias=True)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 扩展卷积核到所有通道
        sobel_x = self.sobel_x.expand(channels, 1, 3, 3)
        sobel_y = self.sobel_y.expand(channels, 1, 3, 3)
        laplacian = self.laplacian.expand(channels, 1, 3, 3)
        center_surround = self.center_surround.expand(channels, 1, 3, 3)
        
        # 计算各种梯度
        grad_x = F.conv2d(x, sobel_x, padding=self.padding, groups=channels)
        grad_y = F.conv2d(x, sobel_y, padding=self.padding, groups=channels)
        grad_lap = F.conv2d(x, laplacian, padding=self.padding, groups=channels)
        grad_cs = F.conv2d(x, center_surround, padding=self.padding, groups=channels)
        grad_idg = self.idg_conv(x)
        
        features = [grad_idg, grad_cs, grad_x, grad_y, grad_lap]
        
        if self.use_vanilla:
            grad_vanilla = self.vanilla_conv(x)
            features.append(grad_vanilla)
        
        # 计算注意力权重
        att_weights = self.channel_attention(x)  # [B, C*num_features, 1, 1]
        
        # 应用注意力权重
        num_features = len(features)
        att_weights = att_weights.view(batch_size, num_features, channels, 1, 1)
        
        # 加权融合
        output = 0
        for i, feat in enumerate(features):
            output = output + feat * att_weights[:, i, :, :, :]
        
        # 最终融合
        output = self.fusion(output)
        
        return output


class MemoryEfficientDGConvModule(nn.Module):
    """内存高效版本，使用gradient checkpoint"""
    def __init__(self, channels, kernel_size=3, use_checkpoint=True, use_vanilla=True):
        super(MemoryEfficientDGConvModule, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.dgconv = OptimizedDGConvModule(channels, kernel_size, use_vanilla)
        
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self.dgconv, x)
        else:
            return self.dgconv(x)


# 为了保持向后兼容，提供原有的轻量级版本
class LightweightDGConvModule(nn.Module):
    """超轻量级版本"""
    def __init__(self, channels, kernel_size=3):
        super(LightweightDGConvModule, self).__init__()
        
        # 使用分离卷积减少参数
        self.horizontal_conv = nn.Conv2d(channels, channels, 
                                       kernel_size=(1, 3), 
                                       padding=(0, 1), 
                                       groups=channels, bias=False)
        self.vertical_conv = nn.Conv2d(channels, channels, 
                                     kernel_size=(3, 1), 
                                     padding=(1, 0), 
                                     groups=channels, bias=False)
        
        # 初始化为梯度算子
        with torch.no_grad():
            self.horizontal_conv.weight.data.fill_(0)
            self.vertical_conv.weight.data.fill_(0)
            
            # 设置梯度检测权重
            for c in range(channels):
                if kernel_size == 3:
                    self.horizontal_conv.weight.data[c, 0, 0, 0] = -1
                    self.horizontal_conv.weight.data[c, 0, 0, 1] = 0
                    self.horizontal_conv.weight.data[c, 0, 0, 2] = 1
                    
                    self.vertical_conv.weight.data[c, 0, 0, 0] = -1
                    self.vertical_conv.weight.data[c, 0, 1, 0] = 0
                    self.vertical_conv.weight.data[c, 0, 2, 0] = 1
        
        # 自适应融合
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 2, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Conv2d(channels, channels, 1, bias=True)
        
    def forward(self, x):
        grad_h = self.horizontal_conv(x)
        grad_v = self.vertical_conv(x)
        
        # 计算注意力权重
        B, C, H, W = x.shape
        att = self.channel_attention(x)  # [B, C*2, 1, 1]
        att_h, att_v = torch.chunk(att, 2, dim=1)
        
        # 加权融合
        output = grad_h * att_h + grad_v * att_v
        output = self.fusion(output)
        
        return output


if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    channels = 64
    height, width = 128, 128
    
    # 创建测试输入
    x = torch.randn(batch_size, channels, height, width)
    
    # 测试不同版本
    print("Testing DGConvModule...")
    dgconv = DGConvModule(channels)
    output = dgconv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in dgconv.parameters())}")
    
    print("\nTesting OptimizedDGConvModule...")
    opt_dgconv = OptimizedDGConvModule(channels)
    opt_output = opt_dgconv(x)
    print(f"Output shape: {opt_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in opt_dgconv.parameters())}")
    
    print("\nTesting LightweightDGConvModule...")
    light_dgconv = LightweightDGConvModule(channels)
    light_output = light_dgconv(x)
    print(f"Output shape: {light_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in light_dgconv.parameters())}")
    
    print("\nAll tests passed!")