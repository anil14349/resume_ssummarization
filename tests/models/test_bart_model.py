"""Tests for BART model."""
import pytest
from unittest.mock import patch, MagicMock
from src.models.bart_model import BartResumeModel


@pytest.fixture
def bart_model():
    """BART model fixture."""
    with patch('src.models.bart_model.BartForConditionalGeneration.from_pretrained') as mock_model:
        with patch('src.models.bart_model.BartTokenizer.from_pretrained') as mock_tokenizer:
            model = BartResumeModel()
            yield model


def test_bart_model_initialization(bart_model):
    """Test BART model initialization."""
    assert bart_model.model_name == "facebook/bart-base"
    assert bart_model.max_length == 512
    assert bart_model.min_length == 100
    assert bart_model.temperature == 0.7


def test_generate_summary_success(bart_model, sample_resume_data):
    """Test successful summary generation."""
    summary = bart_model.generate_summary(sample_resume_data)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "John Doe" in summary
    assert "5 years" in summary
    assert "Python" in summary
    assert "Tech Corp" in summary


def test_generate_summary_empty_data(bart_model, empty_resume_data):
    """Test summary generation with empty data."""
    summary = bart_model.generate_summary(empty_resume_data)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert summary != "Error generating summary."


def test_generate_summary_minimal_data(bart_model, minimal_resume_data):
    """Test summary generation with minimal data."""
    summary = bart_model.generate_summary(minimal_resume_data)
    
    assert isinstance(summary, str)
    assert "Jane Smith" in summary
    assert "2 years" in summary
    assert "Python" in summary
    assert "Startup Inc" in summary


def test_clean_summary(bart_model):
    """Test summary cleaning."""
    dirty_summary = "hi,  this is  a Test.  with extra  spaces.  "
    clean_summary = bart_model._clean_summary(dirty_summary)
    
    assert clean_summary == "Hi, this is a Test. With extra spaces"
    assert "  " not in clean_summary


def test_validate_summary(bart_model):
    """Test summary validation."""
    # Valid summary
    valid_summary = "This is a valid summary with enough words to pass the minimum length check but not too many to fail the maximum length check."
    assert bart_model._validate_summary(valid_summary) is True
    
    # Too short
    short_summary = "Too short."
    assert bart_model._validate_summary(short_summary) is False
    
    # Too long
    long_summary = " ".join(["word"] * 250)
    assert bart_model._validate_summary(long_summary) is False
    
    # Empty
    assert bart_model._validate_summary("") is False


def test_contact_info_formatting(bart_model):
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
    
    summary = bart_model.generate_summary(resume_data)
    assert 'test@example.com' in summary
    assert '123-456-7890' in summary


@pytest.mark.parametrize("error_location", ["model", "tokenizer"])
def test_model_initialization_error(error_location):
    """Test error handling during model initialization."""
    with patch('src.models.bart_model.BartForConditionalGeneration.from_pretrained') as mock_model:
        with patch('src.models.bart_model.BartTokenizer.from_pretrained') as mock_tokenizer:
            if error_location == "model":
                mock_model.side_effect = Exception("Model error")
            else:
                mock_tokenizer.side_effect = Exception("Tokenizer error")
                
            with pytest.raises(Exception):
                BartResumeModel()
