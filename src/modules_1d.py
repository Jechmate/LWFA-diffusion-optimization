import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
  """
  Exponential Moving Average (EMA) class
  helps smooth out the model's parameters during training,
  which can lead to a more stable and better-performing
  model by averaging recent updates with past parameter values.
  """
  def __init__(self, beta):
      super().__init__()
      self.beta = beta
      self.step = 0

  def update_model_average(self, ma_model, current_model):
      for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
          old_weight, up_weight = ma_params.data, current_params.data
          ma_params.data = self.update_average(old_weight, up_weight)

  def update_average(self, old, new):
      if old is None:
          return new
      return old * self.beta + (1 - self.beta) * new

  def step_ema(self, ema_model, model, step_start_ema=2000):
      if self.step < step_start_ema:
          self.reset_parameters(ema_model, model)
          self.step += 1
          return
      self.update_model_average(ema_model, model)
      self.step += 1

  def reset_parameters(self, ema_model, model):
      ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
  def __init__(self,
                channels,
                resolution
                ):
      super(SelfAttention, self).__init__()
      self.channels = channels
      self.resolution = resolution
      self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
      self.ln = nn.LayerNorm([channels])
      self.ff_self = nn.Sequential(
          nn.LayerNorm([channels]),
          nn.Linear(channels, channels),
          nn.GELU(),
          nn.Linear(channels, channels),
      )

  def forward(self, x):
      x = x.view(-1, self.resolution, self.channels)
      x_ln = self.ln(x)
      attention_value, _ = self.mha(x_ln, x_ln, x_ln)
      attention_value = attention_value + x
      attention_value = self.ff_self(attention_value) + attention_value
      # Reshape back to (batch, channels, resolution)
      return attention_value.view(-1, self.channels, self.resolution)

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
      super().__init__()
      self.residual = residual
      if not mid_channels:
          mid_channels = out_channels

      self.double_conv = nn.Sequential(
          nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
          nn.InstanceNorm1d(mid_channels),
          nn.GELU(),
          nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
          nn.InstanceNorm1d(out_channels),
      )


  def forward(self, x):
      if self.residual:
          return F.gelu(x + self.double_conv(x))
      else:
          return self.double_conv(x)

class Down(nn.Module):
  def __init__(self, 
               in_channels, 
               out_channels, 
               emb_dim=256,
               dropout_rate=0.05
              ):
      super().__init__()

      self.maxpool_conv = nn.Sequential(
          nn.MaxPool1d(2),
          DoubleConv(in_channels, in_channels, residual=True),
          DoubleConv(in_channels, out_channels),
          nn.Dropout(p=dropout_rate),
      )

      self.emb_layer = nn.Sequential(
          nn.SiLU(),
          nn.Linear(
              emb_dim,
              out_channels
          ),
          nn.Dropout(p=dropout_rate),
      )

  def forward(self, x, t):
      x = self.maxpool_conv(x)
      emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
      return x + emb

class Up(nn.Module):
  def __init__(self, 
               in_channels, 
               out_channels, 
               emb_dim=256,
               dropout_rate=0.05
              ):
      super().__init__()

      self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
      self.conv = nn.Sequential(
          DoubleConv(in_channels, in_channels, residual=True),
          DoubleConv(in_channels, out_channels, in_channels // 2),
          nn.Dropout(p=dropout_rate),
      )

      self.emb_layer = nn.Sequential(
          nn.SiLU(),
          nn.Linear(
              emb_dim,
              out_channels
          ),
          nn.Dropout(p=dropout_rate),
      )

  def forward(self, x, skip_x, t):
      x = self.up(x)
      x = torch.cat([skip_x, x], dim=1)
      x = self.conv(x)
      emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
      return x + emb


class UNet_conditional(nn.Module):

    def __init__(self,
                 time_dim=256,
                 device="cuda",
                 settings_dim=13,
                 resolution=1024,
                 dropout_rate=0.05,
                 sampler_type="EDM"):
        
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.sampler_type = sampler_type
        
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128, dropout_rate=dropout_rate)
        self.sa1 = SelfAttention(128, resolution // 2)
        self.down2 = Down(128, 256, dropout_rate=dropout_rate)
        self.sa2 = SelfAttention(256, resolution // 4)
        self.down3 = Down(256, 256, dropout_rate=dropout_rate)
        self.sa3 = SelfAttention(256, resolution // 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, dropout_rate=dropout_rate)
        self.sa4 = SelfAttention(128, resolution // 4)
        self.up2 = Up(256, 64, dropout_rate=dropout_rate)
        self.sa5 = SelfAttention(64, resolution // 2)
        self.up3 = Up(128, 64, dropout_rate=dropout_rate)
        self.sa6 = SelfAttention(64, resolution)

        self.outc = nn.Conv1d(64, 1, kernel_size=1)

        self.label_prep = nn.Sequential(
           nn.BatchNorm1d(settings_dim),
           nn.Linear(settings_dim, time_dim),
           nn.SiLU(),
        )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        ).to(self.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Add class embedding to time encoding if class label y is provided
        if y is not None:
            y = self.label_prep(y).squeeze()
            t += y

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        return output

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

class EDMPrecond(torch.nn.Module):
    def __init__(self,
                 resolution      = 1024,                     
                 settings_dim    = 13, # Number of class labels, 0 = unconditional.
                 sigma_min       = 0,  # Minimum supported noise level.
                 sigma_max       = float('inf'), # Maximum supported noise level.
                 sigma_data      = 0.5, # Expected standard deviation of the training data.
                 model_type      = 'UNet_conditional', # Class name of the underlying model.
                 device          = 'cuda',
                 dropout_rate    = 0.05
    ):
        super().__init__()
        self.resolution = resolution
        self.settings_dim = settings_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](time_dim = 256,
                                            device = device,
                                            settings_dim = settings_dim,
                                            resolution = resolution,
                                            dropout_rate = dropout_rate,
                                            sampler_type = "EDM"
                                          )


    def forward(self,
                x,
                sigma,
                settings=None
                ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        settings = None if self.settings_dim == 0 or settings is None else settings.to(torch.float32).reshape(-1, self.settings_dim)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype),
                         c_noise.flatten(),
                         settings)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------