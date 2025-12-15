import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class DiscreteDiffusion(nn.Module):
    def __init__(self, model, vocab_size, mask_token_id, pad_token_id, timesteps=1000):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.timesteps = timesteps
        
    def get_mask_prob(self, t):
        """
        Cosine schedule for masking rate.
        t: [batch_size] or float, between 0 and 1 (normalized time)
        Returns: probability of a token being MASKED at time t.
        """
        # We want mask_prob to go from 0 (at t=0) to 1 (at t=1)
        # Standard cosine schedule: alpha_bar = cos( (t+s)/(1+s) * pi/2 )^2
        # mask_prob = 1 - alpha_bar
        
        s = 0.008
        if isinstance(t, torch.Tensor):
            t = t.clamp(0, 1)
        
        # Angle goes from s/(1+s)*pi/2 to pi/2
        angle = (t + s) / (1 + s) * (math.pi / 2)
        alpha_bar = torch.cos(angle) ** 2
        
        # Normalize so alpha_bar(0) = 1
        alpha_bar_0 = math.cos(s / (1 + s) * (math.pi / 2)) ** 2
        alpha_bar = alpha_bar / alpha_bar_0
        
        return 1 - alpha_bar

    def q_sample(self, x_start, t, mask_strategy='random'):
        """
        Forward diffusion process: Corrupt x_start with masks.
        x_start: [batch, seq_len]
        t: [batch] (int timesteps)
        """
        batch, seq_len = x_start.shape
        device = x_start.device
        
        # Normalize t
        t_norm = t.float() / self.timesteps
        mask_probs = self.get_mask_prob(t_norm).to(device) # [batch]
        
        # Expand for broadcasting
        mask_probs = mask_probs.view(batch, 1)
        
        # Create mask
        if mask_strategy == 'random':
            # Uniform random masking
            mask_mask = torch.rand(batch, seq_len, device=device) < mask_probs
            
        elif mask_strategy == 'span':
            # Vectorized Span Masking (Efficient)
            # Instead of iterative loops, we use max_pool1d to expand random start points.
            
            span_len = 5 # Fixed span length for efficiency
            
            # Probability of a token being the start of a span
            # p_start ~= p_mask / span_len
            prob_start = mask_probs / span_len
            
            # Generate start points [batch, seq_len]
            starts = torch.rand(batch, seq_len, device=device) < prob_start
            
            # Expand using Max Pooling
            # [batch, 1, seq_len]
            starts_float = starts.float().unsqueeze(1)
            
            # Max pool smears the 1s to create spans
            # kernel_size=span_len, stride=1, padding=span_len//2 centers the span
            mask_float = F.max_pool1d(
                starts_float, 
                kernel_size=span_len, 
                stride=1, 
                padding=span_len // 2
            )
            
            mask_mask = mask_float.squeeze(1) > 0
            
            # Ensure size matches (padding might cause slight mismatch)
            mask_mask = mask_mask[:, :seq_len]
        
        else:
            raise ValueError(f"Unknown mask strategy: {mask_strategy}")

        # Don't mask padding tokens!
        not_pad = x_start != self.pad_token_id
        mask_mask = mask_mask & not_pad
        
        # Apply mask
        x_t = x_start.clone()
        x_t[mask_mask] = self.mask_token_id
        
        return x_t, mask_mask

    def compute_loss(self, x_start):
        """
        Computes the diffusion loss (Cross Entropy on masked tokens).
        """
        batch, seq_len = x_start.shape
        device = x_start.device
        
        # 1. Sample timestep t
        t = torch.randint(1, self.timesteps + 1, (batch,), device=device)
        
        # 2. q_sample (Add noise)
        # Use 'span' strategy randomly to encourage robustness
        # Use numpy for boolean choice to avoid GPU sync
        strategy = 'span' if np.random.rand() < 0.5 else 'random'
        x_t, mask_mask = self.q_sample(x_start, t, mask_strategy=strategy)
        
        # 3. Predict
        logits = self.model(x_t) # [batch, seq_len, vocab_size]
        
        # 4. Loss
        # Only compute loss on MASKED tokens
        # Flatten
        logits = logits.view(-1, self.vocab_size)
        targets = x_start.view(-1)
        mask_flat = mask_mask.view(-1)
        
        if mask_flat.sum() == 0:
            # Edge case: no tokens masked (t=0 or very small)
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        loss = F.cross_entropy(logits[mask_flat], targets[mask_flat])
        
        return loss

    @torch.no_grad()
    def sample(self, shape, steps=None, temperature=1.0):
        """
        Generate samples from pure noise.
        shape: (batch, seq_len)
        """
        device = next(self.model.parameters()).device
        batch, seq_len = shape
        if steps is None:
            steps = self.timesteps
            
        # Start with all MASKs
        x_t = torch.full(shape, self.mask_token_id, device=device)
        
        # Iterative Denoising
        # We use a simplified "Mask-Predict" style sampling (like VQ-Diffusion / Muse)
        # At each step, we predict x_0, then re-mask a smaller portion.
        
        # Schedule for sampling
        # We go from t=T (100% masked) to t=0 (0% masked)
        times = torch.linspace(self.timesteps, 0, steps + 1, device=device).long()
        
        for i in range(steps):
            t_curr = times[i]
            t_next = times[i+1]
            
            # 1. Predict x_0 from x_t
            logits = self.model(x_t)
            
            # Sample from logits with temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            probs = F.softmax(logits, dim=-1)
            
            # Sample predicted tokens
            x_0_pred = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch, seq_len)
            
            # 2. Decide which tokens to keep
            # We want to keep tokens where the model is "confident"
            # Confidence = probability of the selected token
            confidence = torch.gather(probs, -1, x_0_pred.unsqueeze(-1)).squeeze(-1)
            
            # 3. Re-mask for next step
            # Calculate how many tokens should be masked at t_next
            mask_prob_next = self.get_mask_prob(t_next.float() / self.timesteps)
            num_to_mask = (mask_prob_next * seq_len).long().clamp(min=0)
            
            # We keep the (seq_len - num_to_mask) most confident tokens
            # and mask the rest.
            
            # Add noise to confidence to prevent getting stuck? (Gumbel noise)
            # gumbel = -torch.log(-torch.log(torch.rand_like(confidence) + 1e-9) + 1e-9)
            # scores = confidence + gumbel
            scores = confidence # Pure confidence works well for "greedy" masking
            
            # Identify tokens to mask
            # We want to mask the ones with LOWEST score
            # So we find the threshold
            
            # Create next x_t
            x_next = x_0_pred.clone()
            
            if num_to_mask > 0:
                # Find indices of lowest scores
                # topk returns highest, so we use negative scores or sort
                # We want to mask the bottom 'num_to_mask'
                _, indices = torch.topk(scores, k=num_to_mask, dim=1, largest=False)
                
                # Create mask
                mask = torch.zeros_like(x_next, dtype=torch.bool)
                mask.scatter_(1, indices, True)
                
                x_next[mask] = self.mask_token_id
            
            # Update
            x_t = x_next
            
        return x_t

    @torch.no_grad()
    def infill(self, input_ids, mask_mask, steps=None, temperature=1.0):
        """
        Infill/Continue a sequence.
        input_ids: [batch, seq_len] - The sequence with known tokens.
        mask_mask: [batch, seq_len] - True where we want to generate, False where we keep fixed.
        """
        device = input_ids.device
        batch, seq_len = input_ids.shape
        if steps is None:
            steps = self.timesteps
            
        # Start: Known tokens are kept, unknown are MASKed
        x_t = input_ids.clone()
        x_t[mask_mask] = self.mask_token_id
        
        times = torch.linspace(self.timesteps, 0, steps + 1, device=device).long()
        
        for i in range(steps):
            t_curr = times[i]
            t_next = times[i+1]
            
            # 1. Predict x_0
            logits = self.model(x_t)
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            x_0_pred = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch, seq_len)
            
            # 2. Confidence
            confidence = torch.gather(probs, -1, x_0_pred.unsqueeze(-1)).squeeze(-1)
            
            # 3. Re-mask
            # Only re-mask within the 'mask_mask' region!
            # We want to gradually reveal tokens in the masked region.
            
            mask_prob_next = self.get_mask_prob(t_next.float() / self.timesteps)
            
            # Total tokens to be masked at next step (global)
            # But we only care about the region we are generating.
            # Let's say we want to keep (1 - mask_prob) of the *target* region.
            
            # Number of tokens to mask *within the target region*
            num_target_tokens = mask_mask.sum(dim=1, keepdim=True) # [batch, 1]
            num_to_mask = (mask_prob_next * num_target_tokens).long()
            
            x_next = x_0_pred.clone()
            
            # Force known tokens
            x_next[~mask_mask] = input_ids[~mask_mask]
            
            # Apply re-masking only to target region
            if num_to_mask.max() > 0:
                # Set confidence of fixed tokens to infinity so they are never masked
                scores = confidence.clone()
                scores[~mask_mask] = float('inf')
                
                # We want to mask the lowest scores
                # But since fixed tokens are inf, they won't be picked unless everything is inf
                
                # We need to mask 'num_to_mask' tokens per batch item
                # torch.topk doesn't support per-row k if k varies.
                # Simplified: assume k is roughly same or use loop.
                # For batch=1 (inference), it's easy.
                
                for b in range(batch):
                    k = num_to_mask[b].item()
                    if k > 0:
                        # Get scores for this batch
                        b_scores = scores[b]
                        # Find k lowest
                        _, indices = torch.topk(b_scores, k=k, largest=False)
                        x_next[b, indices] = self.mask_token_id
            
            x_t = x_next
            
        return x_t
