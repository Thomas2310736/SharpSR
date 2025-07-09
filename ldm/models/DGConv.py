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


# ============ 标准版本 DGConvModule with AIIBlock ============
class DGConvModule(nn.Module):
    """标准版自适应方向梯度卷积模块 - 集成AIIBlock"""
    def __init__(self, in_channels, out_channels=None, kernel_size=3, 
                 use_vanilla=True, reduction_ratio=4, use_aiiblock=True):
        super(DGConvModule, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_vanilla = use_vanilla
        self.use_aiiblock = use_aiiblock
        
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
        
        # AIIBlock集成
        if use_aiiblock:
            self.aiiblock = StandardAIIBlock(out_channels, reduction_ratio)
        
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
        
        # 应用AIIBlock
        if self.use_aiiblock:
            output = self.aiiblock(output, x)
        
        return output


# ============ 优化版本 OptimizedDGConvModule with AIIBlock ============
class OptimizedDGConvModule(nn.Module):
    """优化版本的DGConv - 显存效率更高 - 集成AIIBlock"""
    def __init__(self, channels, kernel_size=3, use_vanilla=True, use_aiiblock=True):
        super(OptimizedDGConvModule, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_vanilla = use_vanilla
        self.use_aiiblock = use_aiiblock
        
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
        
        # AIIBlock集成
        if use_aiiblock:
            self.aiiblock = OptimizedAIIBlock(channels)
        
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
        
        # 应用AIIBlock
        if self.use_aiiblock:
            output = self.aiiblock(output, x)
        
        return output


# ============ 轻量级版本 LightweightDGConvModule with AIIBlock ============
class LightweightDGConvModule(nn.Module):
    """超轻量级版本 - 集成AIIBlock"""
    def __init__(self, channels, kernel_size=3, use_aiiblock=True):
        super(LightweightDGConvModule, self).__init__()
        self.use_aiiblock = use_aiiblock
        
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
        
        # AIIBlock集成
        if use_aiiblock:
            self.aiiblock = LightweightAIIBlock(channels)
        
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
        
        # 应用AIIBlock
        if self.use_aiiblock:
            output = self.aiiblock(output, x)
        
        return output


# ============ AIIBlock的三个版本 ============
class StandardAIIBlock(nn.Module):
    """标准版AIIBlock - 用于DGConvModule"""
    def __init__(self, channels, reduction=4):
        super(StandardAIIBlock, self).__init__()
        self.channels = channels
        
        # 对比度分支 - 使用CSA和普通卷积
        self.contrast_branch = nn.Sequential(
            CenterSurroundingAggregationConv(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 交互注意力机制
        self.interaction_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, gradient_feat, original_input):
        # 对比度特征提取
        contrast_feat = self.contrast_branch(original_input)
        
        # 连接特征
        concat_feat = torch.cat([gradient_feat, contrast_feat], dim=1)
        
        # 计算交互注意力
        attention = self.interaction_attention(concat_feat)
        
        # 应用注意力
        weighted_feat = concat_feat * attention
        
        # 融合特征
        fused = self.fusion(weighted_feat)
        
        # 残差连接
        output = fused + gradient_feat
        
        return output


class OptimizedAIIBlock(nn.Module):
    """优化版AIIBlock - 用于OptimizedDGConvModule"""
    def __init__(self, channels, reduction=4):
        super(OptimizedAIIBlock, self).__init__()
        self.channels = channels
        
        # 轻量级对比度分支
        self.register_buffer('contrast_kernel', torch.tensor([
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]]
        ], dtype=torch.float32).unsqueeze(0) / 9.0)
        
        self.contrast_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.contrast_norm = nn.BatchNorm2d(channels)
        self.contrast_relu = nn.ReLU(inplace=True)
        
        # 简化的交互注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, gradient_feat, original_input):
        B, C, H, W = gradient_feat.shape
        
        # 对比度特征提取（使用深度可分离卷积）
        contrast_kernel = self.contrast_kernel.expand(C, 1, 3, 3)
        contrast_feat = F.conv2d(original_input, contrast_kernel, padding=1, groups=C)
        contrast_feat = self.contrast_conv(contrast_feat)
        contrast_feat = self.contrast_norm(contrast_feat)
        contrast_feat = self.contrast_relu(contrast_feat)
        
        # 空间注意力
        avg_feat = torch.mean(gradient_feat, dim=1, keepdim=True)
        max_feat, _ = torch.max(gradient_feat, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_feat, max_feat], dim=1))
        
        # 通道注意力
        channel_att = self.channel_attention(gradient_feat)
        
        # 应用注意力
        enhanced_gradient = gradient_feat * spatial_att * channel_att
        enhanced_contrast = contrast_feat * (1 - spatial_att) * channel_att
        
        # 融合
        output = enhanced_gradient + enhanced_contrast + gradient_feat
        
        return output


class LightweightAIIBlock(nn.Module):
    """轻量级AIIBlock - 用于LightweightDGConvModule"""
    def __init__(self, channels, reduction=8):
        super(LightweightAIIBlock, self).__init__()
        self.channels = channels
        
        # 简单的1x1卷积用于特征变换
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 简单的残差缩放
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        
    def forward(self, gradient_feat, original_input):
        # 计算简单的注意力权重
        attention = self.transform(gradient_feat)
        
        # 应用注意力并添加残差
        output = gradient_feat + self.alpha * (gradient_feat * attention)
        
        return output


# ============ 内存高效版本（带gradient checkpoint） ============
class MemoryEfficientDGConvModule(nn.Module):
    """内存高效版本，使用gradient checkpoint"""
    def __init__(self, channels, kernel_size=3, use_checkpoint=True, 
                 use_vanilla=True, use_aiiblock=True, dgconv_type='optimized'):
        super(MemoryEfficientDGConvModule, self).__init__()
        self.use_checkpoint = use_checkpoint
        
        # 根据类型选择对应的DGConv
        if dgconv_type == 'standard':
            self.dgconv = DGConvModule(channels, channels, kernel_size, 
                                      use_vanilla=use_vanilla, use_aiiblock=use_aiiblock)
        elif dgconv_type == 'optimized':
            self.dgconv = OptimizedDGConvModule(channels, kernel_size, 
                                               use_vanilla=use_vanilla, use_aiiblock=use_aiiblock)
        elif dgconv_type == 'lightweight':
            self.dgconv = LightweightDGConvModule(channels, kernel_size, use_aiiblock=use_aiiblock)
        else:
            raise ValueError(f"Unknown dgconv_type: {dgconv_type}")
        
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self.dgconv, x)
        else:
            return self.dgconv(x)


# ============ 为StableSR设计的专用模块 ============
class DGConvResBlock(nn.Module):
    """为StableSR设计的增强型残差块，集成DGConv和AIIBlock"""
    def __init__(self, channels, kernel_size=3, use_dgconv=True, 
                 dgconv_type='optimized', use_aiiblock=True):
        super(DGConvResBlock, self).__init__()
        self.use_dgconv = use_dgconv
        
        if use_dgconv:
            if dgconv_type == 'standard':
                self.conv1 = DGConvModule(channels, channels, kernel_size, use_aiiblock=use_aiiblock)
                self.conv2 = DGConvModule(channels, channels, kernel_size, use_aiiblock=False)
            elif dgconv_type == 'optimized':
                self.conv1 = OptimizedDGConvModule(channels, kernel_size, use_aiiblock=use_aiiblock)
                self.conv2 = OptimizedDGConvModule(channels, kernel_size, use_aiiblock=False)
            elif dgconv_type == 'lightweight':
                self.conv1 = LightweightDGConvModule(channels, kernel_size, use_aiiblock=use_aiiblock)
                self.conv2 = LightweightDGConvModule(channels, kernel_size, use_aiiblock=False)
            else:
                raise ValueError(f"Unknown dgconv_type: {dgconv_type}")
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=False)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=False)
        
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class StableSREncoderBlock(nn.Module):
    """专门为StableSR设计的编码器块，包含DGConv和AIIBlock"""
    def __init__(self, in_channels, out_channels, stride=1, 
                 use_dgconv=True, use_aiiblock=True, dgconv_type='optimized'):
        super(StableSREncoderBlock, self).__init__()
        
        # 下采样层（如果需要）
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
            
        # 主要卷积层
        if use_dgconv:
            if dgconv_type == 'standard':
                conv_module = lambda c: DGConvModule(c, c, kernel_size=3, use_aiiblock=use_aiiblock)
            elif dgconv_type == 'optimized':
                conv_module = lambda c: OptimizedDGConvModule(c, kernel_size=3, use_aiiblock=use_aiiblock)
            else:
                conv_module = lambda c: LightweightDGConvModule(c, kernel_size=3, use_aiiblock=use_aiiblock)
        else:
            conv_module = lambda c: nn.Conv2d(c, c, 3, padding=1, bias=False)
            
        # 第一个卷积块（带AIIBlock）
        self.conv1 = conv_module(out_channels if self.downsample else in_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二个卷积块（不带AIIBlock以节省资源）
        if use_dgconv:
            if dgconv_type == 'standard':
                self.conv2 = DGConvModule(out_channels, out_channels, kernel_size=3, use_aiiblock=False)
            elif dgconv_type == 'optimized':
                self.conv2 = OptimizedDGConvModule(out_channels, kernel_size=3, use_aiiblock=False)
            else:
                self.conv2 = LightweightDGConvModule(out_channels, kernel_size=3, use_aiiblock=False)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 处理下采样
        if self.downsample is not None:
            residual = self.downsample(x)
            out = self.conv1(self.downsample[0](x))  # 只使用conv进行特征提取
        else:
            residual = x
            out = self.conv1(x)
            
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out = out + residual
        out = self.relu2(out)
            
        return out


if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    channels = 64
    height, width = 128, 128
    
    # 创建测试输入
    x = torch.randn(batch_size, channels, height, width)
    
    print("=" * 50)
    print("Testing all DGConv variants with AIIBlock")
    print("=" * 50)
    
    # 测试标准版本
    print("\n1. Testing Standard DGConvModule with AIIBlock...")
    dgconv_standard = DGConvModule(channels, use_aiiblock=True)
    output_standard = dgconv_standard(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output_standard.shape}")
    print(f"   Parameters: {sum(p.numel() for p in dgconv_standard.parameters()):,}")
    
    # 测试优化版本
    print("\n2. Testing Optimized DGConvModule with AIIBlock...")
    dgconv_optimized = OptimizedDGConvModule(channels, use_aiiblock=True)
    output_optimized = dgconv_optimized(x)
    print(f"   Output shape: {output_optimized.shape}")
    print(f"   Parameters: {sum(p.numel() for p in dgconv_optimized.parameters()):,}")
    
    # 测试轻量级版本
    print("\n3. Testing Lightweight DGConvModule with AIIBlock...")
    dgconv_lightweight = LightweightDGConvModule(channels, use_aiiblock=True)
    output_lightweight = dgconv_lightweight(x)
    print(f"   Output shape: {output_lightweight.shape}")
    print(f"   Parameters: {sum(p.numel() for p in dgconv_lightweight.parameters()):,}")
    
    # 测试内存高效版本
    print("\n4. Testing Memory Efficient versions...")
    for dgconv_type in ['standard', 'optimized', 'lightweight']:
        mem_efficient = MemoryEfficientDGConvModule(
            channels, dgconv_type=dgconv_type, use_aiiblock=True
        )
        output_mem = mem_efficient(x)
        print(f"   {dgconv_type.capitalize()} - Output shape: {output_mem.shape}")
    
    # 测试StableSR专用模块
    print("\n5. Testing StableSR-specific modules...")
    
    # 测试残差块
    print("   Testing DGConvResBlock...")
    for dgconv_type in ['standard', 'optimized', 'lightweight']:
        res_block = DGConvResBlock(channels, dgconv_type=dgconv_type, use_aiiblock=True)
        res_output = res_block(x)
        print(f"   {dgconv_type.capitalize()} ResBlock - Output: {res_output.shape}")
    
    # 测试编码器块
    print("\n   Testing StableSREncoderBlock...")
    # 测试无下采样
    encoder_block1 = StableSREncoderBlock(channels, channels, use_dgconv=True, 
                                         use_aiiblock=True, dgconv_type='optimized')
    enc_output1 = encoder_block1(x)
    print(f"   No downsampling - Output: {enc_output1.shape}")
    
    # 测试有下采样
    x_large = torch.randn(batch_size, channels, 256, 256)
    encoder_block2 = StableSREncoderBlock(channels, channels*2, stride=2, 
                                         use_dgconv=True, use_aiiblock=True,
                                         dgconv_type='lightweight')
    enc_output2 = encoder_block2(x_large)
    print(f"   With downsampling - Input: {x_large.shape}, Output: {enc_output2.shape}")
    
    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)