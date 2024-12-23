"""Tests for T5 model."""
import pytest
from unittest.mock import patch, MagicMock
from src.models.t5_model import T5ResumeModel


@pytest.fixture
def t5_model():
    """T5 model fixture."""
    with patch('src.models.t5_model.T5ForConditionalGeneration.from_pretrained') as mock_model:
        with patch('src.models.t5_model.T5Tokenizer.from_pretrained') as mock_tokenizer:
            model = T5ResumeModel()
            yield model


def test_t5_model_initialization(t5_model):
    """Test T5 model initialization."""
    assert t5_model.model_name == "t5-base"
    assert t5_model.max_length == 512
    assert t5_model.min_length == 100
    assert t5_model.temperature == 0.7


def test_generate_summary_success(t5_model, sample_resume_data):
    """Test successful summary generation."""
    summary = t5_model.generate_summary(sample_resume_data)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "John Doe" in summary
    assert "5 years" in summary
    assert "Python" in summary
    assert "Tech Corp" in summary


def test_generate_summary_empty_data(t5_model, empty_resume_data):
    """Test summary generation with empty data."""
    summary = t5_model.generate_summary(empty_resume_data)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert summary != "Error generating summary."


def test_generate_summary_minimal_data(t5_model, minimal_resume_data):
    """Test summary generation with minimal data."""
    summary = t5_model.generate_summary(minimal_resume_data)
    
    assert isinstance(summary, str)
    assert "Jane Smith" in summary
    assert "2 years" in summary
    assert "Python" in summary
    assert "Startup Inc" in summary


def test_clean_summary(t5_model):
    """Test summary cleaning."""
    dirty_summary = "hi,  this is  a Test.  with extra  spaces.  "
    clean_summary = t5_model._clean_summary(dirty_summary)
    
    assert clean_summary == "Hi, this is a Test. With extra spaces"
    assert "  " not in clean_summary


def test_validate_summary(t5_model):
    """Test summary validation."""
    # Valid summary
    valid_summary = "This is a valid summary with enough words to pass the minimum length check but not too many to fail the maximum length check."
    assert t5_model._validate_summary(valid_summary) is True
    
    # Too short
    short_summary = "Too short."
    assert t5_model._validate_summary(short_summary) is False
    
    # Too long
    long_summary = " ".join(["word"] * 250)
    assert t5_model._validate_summary(long_summary) is False
    
    # Empty
    assert t5_model._validate_summary("") is False


def test_contact_info_formatting(t5_model):
    """Test contact information formatting in summary."""
    resume_data = {
        'name': 'Test User',
        'years_experience': 3,
        'skills': ['Python'],
        'companies': ['Test Corp'],
        'achievements': [],
        'contact_info': {
            'email': 'test@example.com',
            'phone': '123-456-7890'
        }
    }
    
    summary = t5_model.generate_summary(resume_data)
    assert 'test@example.com' in summary
    assert '123-456-7890' in summary


@pytest.mark.parametrize("error_location", ["model", "tokenizer"])
def test_model_initialization_error(error_location):
    """Test error handling during model initialization."""
    with patch('src.models.t5_model.T5ForConditionalGeneration.from_pretrained') as mock_model:
        with patch('src.models.t5_model.T5Tokenizer.from_pretrained') as mock_tokenizer:
            if error_location == "model":
                mock_model.side_effect = Exception("Model error")
            else:
                mock_tokenizer.side_effect = Exception("Tokenizer error")
                
            with pytest.raises(Exception):
                T5ResumeModel()
