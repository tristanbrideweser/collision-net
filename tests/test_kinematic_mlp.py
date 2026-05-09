import torch
import pytest
from src.model.encoders.kinematic_mlp import CartesianConfigEncoder

# --- Fixtures ---
# Fixtures allow you to define reusable variables for your tests.
@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def num_keypoints():
    # 9 keypoints for the Franka arm (base -> hand)
    return 9

@pytest.fixture
def out_dim():
    return 1024

@pytest.fixture
def encoder(out_dim):
    return CartesianConfigEncoder(out_dim=out_dim)

# --- Tests ---

def test_encoder_output_shape(encoder, batch_size, num_keypoints, out_dim):
    """Verify the encoder outputs the correct tensor dimensions."""
    dummy_input = torch.randn(batch_size, num_keypoints, 3)
    output = encoder(dummy_input)
    
    expected_shape = (batch_size, num_keypoints, out_dim)
    assert output.shape == expected_shape, \
        f"Expected shape {expected_shape}, but got {output.shape}"

def test_encoder_gradient_flow(encoder, batch_size, num_keypoints, out_dim):
    """Verify that gradients successfully flow backward through the network."""
    dummy_input = torch.randn(batch_size, num_keypoints, 3, requires_grad=True)
    
    # Forward pass
    output = encoder(dummy_input)
    
    # Create a dummy loss (e.g., mean of all outputs) and trigger backprop
    loss = output.mean()
    loss.backward()
    
    # Check that gradients exist for the network parameters
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient computed for {name}"
        assert torch.sum(torch.abs(param.grad)) > 0, f"Gradient is zero for {name}"

def test_encoder_device_compatibility(encoder, batch_size, num_keypoints):
    """Verify the model can successfully run on an available accelerator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = encoder.to(device)
    dummy_input = torch.randn(batch_size, num_keypoints, 3).to(device)
    
    try:
        output = encoder(dummy_input)
        assert output.device == device, "Output tensor is on the wrong device"
    except Exception as e:
        pytest.fail(f"Forward pass failed on device {device} with error: {e}")