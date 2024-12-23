"""Tests for model factory."""
import pytest
from unittest.mock import patch
from src.models.model_factory import create_model
from src.models.gpt2_model import GPT2Model
from src.models.t5_model import T5ResumeModel
from src.models.bart_model import BartResumeModel


def test_create_gpt2_model():
    """Test GPT-2 model creation."""
    with patch('src.models.gpt2_model.GPT2Model.__init__', return_value=None):
        model = create_model("gpt2")
        assert isinstance(model, GPT2Model)


def test_create_t5_model():
    """Test T5 model creation."""
    with patch('src.models.t5_model.T5ResumeModel.__init__', return_value=None):
        model = create_model("t5")
        assert isinstance(model, T5ResumeModel)


def test_create_bart_model():
    """Test BART model creation."""
    with patch('src.models.bart_model.BartResumeModel.__init__', return_value=None):
        model = create_model("bart")
        assert isinstance(model, BartResumeModel)


def test_create_invalid_model():
    """Test invalid model type handling."""
    with pytest.raises(ValueError, match="Invalid model type"):
        create_model("invalid_model")


@pytest.mark.parametrize("model_type,expected_class", [
    ("gpt2", GPT2Model),
    ("t5", T5ResumeModel),
    ("bart", BartResumeModel)
])
def test_model_type_mapping(model_type, expected_class):
    """Test model type to class mapping."""
    with patch(f'src.models.{model_type}_model.{expected_class.__name__}.__init__', return_value=None):
        model = create_model(model_type)
        assert isinstance(model, expected_class)
