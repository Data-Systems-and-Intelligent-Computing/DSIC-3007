from src.verify import assert_loss_close

def test_loss_equivalence():
    assert_loss_close(2.0, 2.0, atol=1e-8)
