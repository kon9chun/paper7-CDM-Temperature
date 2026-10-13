import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalEmbedding(nn.Module):
    """条件向量嵌入层"""
    def __init__(self, input_dim: int = 11, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.mlp(context)


class ResidualBlock(nn.Module):
    """残差块，包含时间步和条件嵌入"""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = 256, cond_emb_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)

        # 添加时间步嵌入
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb

        # 添加条件嵌入
        cond_emb = self.cond_mlp(cond_emb)
        cond_emb = cond_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + cond_emb

        h = self.block2(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """注意力块"""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(batch_size, channels, height, width)
        out = self.proj(out)

        return x + out


class Downsample(nn.Module):
    """下采样层"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """上采样层"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """条件UNet扩散模型"""
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        cond_emb_dim: int = 256,
        cond_input_dim: int = 11,
        attention_resolutions: Tuple[int, ...] = (16, 8)
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 条件嵌入
        self.cond_embedding = ConditionalEmbedding(cond_input_dim, cond_emb_dim)

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = base_channels
        input_block_channels = [channels]

        # 构建下采样路径
        for i, mult in enumerate(channel_mults):
            out_channels_mult = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    nn.ModuleList([
                        ResidualBlock(channels, out_channels_mult, time_emb_dim, cond_emb_dim),
                        nn.Identity()  # 暂时移除注意力机制以简化
                    ])
                )
                channels = out_channels_mult
                input_block_channels.append(channels)

            if i != len(channel_mults) - 1:
                self.down_samples.append(Downsample(channels))
                input_block_channels.append(channels)

        # 中间块
        self.mid_block1 = ResidualBlock(channels, channels, time_emb_dim, cond_emb_dim)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResidualBlock(channels, channels, time_emb_dim, cond_emb_dim)

        # 上采样路径
        self.up_blocks = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels_mult = base_channels * mult

            for j in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()

                self.up_blocks.append(
                    nn.ModuleList([
                        ResidualBlock(channels + skip_channels, out_channels_mult, time_emb_dim, cond_emb_dim),
                        nn.Identity(),  # 暂时移除注意力机制以简化
                        Upsample(out_channels_mult) if j == num_res_blocks and i != 0 else nn.Identity()
                    ])
                )
                channels = out_channels_mult

        # 最终层
        self.final_norm = nn.GroupNorm(32, channels)
        self.final_conv = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # 检查输入通道数
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input has {x.shape[1]} channels, but UNet expects {self.in_channels} channels. "
                f"Input shape: {x.shape}"
            )

        # 嵌入时间步和条件
        time_emb = self.time_mlp(t)
        cond_emb = self.cond_embedding(context)

        # 初始卷积
        h = self.init_conv(x)
        hs = [h]

        # 下采样路径
        down_block_index = 0
        for i, mult in enumerate(self.channel_mults):
            # 残差块
            for _ in range(self.num_res_blocks):
                block_list = self.down_blocks[down_block_index]
                res_block, attn_block = block_list
                h = res_block(h, time_emb, cond_emb)
                h = attn_block(h)
                hs.append(h)
                down_block_index += 1

            # 下采样（除了最后一层）
            if i != len(self.channel_mults) - 1:
                h = self.down_samples[i](h)
                hs.append(h)

        # 中间块
        h = self.mid_block1(h, time_emb, cond_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb, cond_emb)

        # 上采样路径
        for block_list in self.up_blocks:
            res_block, attn_block, upsample = block_list

            # 获取跳跃连接的特征图
            skip = hs.pop()

            # 如果尺寸不匹配，调整跳跃连接的尺寸
            if h.shape[-2:] != skip.shape[-2:]:
                # 使用插值调整尺寸
                skip = F.interpolate(skip, size=h.shape[-2:], mode='nearest')

            h = torch.cat([h, skip], dim=1)
            h = res_block(h, time_emb, cond_emb)
            h = attn_block(h)
            h = upsample(h)

        # 最终输出
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)

        return h


class DiffusionModel:
    """扩散模型包装类，包含训练和推理功能"""
    def __init__(
        self,
        image_size: Tuple[int, int] = (40, 240),
        channels: int = 1,
        cond_dim: int = 11,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.image_size = image_size
        self.channels = channels
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.device = device

        # 创建UNet模型
        self.model = UNet(
            in_channels=channels,
            out_channels=channels,
            cond_input_dim=cond_dim
        ).to(device)

        # 设置扩散参数
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            # Cosine schedule
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = self.betas.to(device)

        # 预计算扩散参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散过程：在时间步t添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """计算扩散模型的损失"""
        noise = torch.randn_like(x_start)

        # 添加噪声
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 预测噪声
        predicted_noise = self.model(x_noisy, t, context)

        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor, t_index: int) -> torch.Tensor:
        """从p(x_{t-1}|x_t)采样"""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)

        # 预测噪声
        model_output = self.model(x, t, context)

        # 计算x0的预测值
        pred_xstart = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
        pred_xstart = torch.clamp(pred_xstart, -1., 1.)

        # 计算均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, context: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """从纯噪声开始采样生成图像"""
        shape = (batch_size, self.channels, self.image_size[0], self.image_size[1])

        # 从纯噪声开始
        img = torch.randn(shape, device=self.device)

        # 反向扩散过程
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, context, i)

        return img

    def train_step(self, x_batch: torch.Tensor, context_batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """执行一个训练步骤"""
        self.model.train()

        batch_size = x_batch.shape[0]

        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

        # 计算损失
        loss = self.p_losses(x_batch, context_batch, t)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def generate(self, context: torch.Tensor, num_samples: int = 1) -> np.ndarray:
        """生成图像（推理接口）"""
        self.model.eval()

        if len(context.shape) == 1:
            context = context.unsqueeze(0)
        if context.shape[0] != num_samples:
            context = context.repeat(num_samples, 1)

        context = context.to(self.device)

        # 采样生成图像
        samples = self.sample(context, num_samples)

        # 转换为numpy数组
        samples = samples.cpu().numpy()

        # 对于训练好的模型，样本应该在[-1, 1]范围内
        # 对于未训练的模型，我们进行裁剪以确保在合理范围内
        # samples = np.clip(samples, -1.0, 1.0)

        # 归一化到[0, 1]用于显示
        # samples = (samples + 1) / 2

        return samples

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'image_size': self.image_size,
                'channels': self.channels,
                'cond_dim': self.cond_dim,
                'timesteps': self.timesteps
            }
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")