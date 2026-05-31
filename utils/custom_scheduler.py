
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from typing import List, Optional, Union


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """
    Custom scheduler for ZImage training adapted from FLUX reference implementation.
    Adds set_train_timesteps method for training-time sampling.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000

            # sigma sqrt weighing is significantly higher at the end and lower at the beginning
            sigma_sqrt_weighing = (self.sigmas**-2.0).float()
            # clip at 1e4 (1e6 is too high)
            sigma_sqrt_weighing = torch.clamp(sigma_sqrt_weighing, max=1e4)
            # bring to a mean of 1
            sigma_sqrt_weighing = sigma_sqrt_weighing / sigma_sqrt_weighing.mean()

            # Create linear timesteps from 1000 to 0
            timesteps = torch.linspace(1000, 0, num_timesteps, device="cpu")

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = sigma_sqrt_weighing
            
            self.use_dynamic_shifting = False

    def get_weights_for_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get the weights for the timesteps"""
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item() for t in timesteps]

        # Get the weights for the timesteps
        weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        """Get sigmas for given timesteps"""
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise according to flow matching.
        zt = (1 - t) * x + t * z1
        """
        # timestep needs to be in [0, 1], we store them in [0, 1000]
        # noisy_sample = (1 - timestep) * latent + timestep * noise
        t_01 = (timesteps / 1000).to(original_samples.device)
        
        # Ensure proper broadcasting
        while len(t_01.shape) < len(original_samples.shape):
            t_01 = t_01.unsqueeze(-1)
        
        noisy_model_input = (1 - t_01) * original_samples + t_01 * noise
        
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """Scale model input (no-op for flow matching)"""
        return sample

    def set_train_timesteps(self, num_timesteps, device, linear=False, mu=None):
        """
        Set timesteps for training.
        """
        if linear:
            # Linear spacing from 1000 to 0
            timesteps = torch.linspace(1000, 0, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        else:
            # Distribute timesteps closer to center
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = (1 - t) * 1000

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)

            return timesteps
