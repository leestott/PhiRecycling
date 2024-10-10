# test for flash_attn status
import torch
import flash_attn

# Verify installation
print(flash_attn.__version__)
print(torch.cuda.is_available())

# Test scaled dot product attention
q = torch.randn(10, 64, device="cuda")
k = torch.randn(10, 64, device="cuda")
v = torch.randn(10, 64, device="cuda")

output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
print(output)