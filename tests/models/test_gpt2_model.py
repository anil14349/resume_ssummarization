"""Tests for GPT-2 model."""
import re
import random
import pytest
from unittest.mock import patch, MagicMock
from src.models.gpt2_model import GPT2Model


@pytest.fixture
def gpt2_model():
    """GPT-2 model fixture."""
    with patch('src.models.gpt2_model.GPT2LMHeadModel.from_pretrained') as mock_model:
        with patch('src.models.gpt2_model.GPT2Tokenizer.from_pretrained') as mock_tokenizer:
            model = GPT2Model()
            yield model


@pytest.fixture
def sample_resume_data():
    """Sample resume data fixture."""
    return {
        "name": "john doe",
        "years_experience": 5,
        "skills": ["Python", "Machine Learning", "Data Science"],
        "companies": ["Tech Corp", "Innovation Inc"],
        "achievements": ["Led a team of 10 engineers", "Increased efficiency by 40%"],
        "contact_info": {
            "email": "john@example.com",
            "phone": "123-456-7890"
        }
    }


@pytest.fixture
def empty_resume_data():
    """Empty resume data fixture."""
    return {}


@pytest.fixture
def minimal_resume_data():
    """Minimal resume data fixture."""
    return {
        "name": "jane smith",
        "years_experience": 2,
        "skills": ["Python"],
        "companies": ["Startup Inc"]
    }


def test_gpt2_model_initialization(gpt2_model):
    """Test GPT-2 model initialization."""
    assert gpt2_model.model_name == "gpt2"
    assert gpt2_model.max_length == 512
    assert gpt2_model.min_length == 100
    assert gpt2_model.temperature == 0.7


def test_generate_summary_success(gpt2_model, sample_resume_data):
    """Test successful summary generation."""
    summary = gpt2_model.generate_summary(sample_resume_data)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "John Doe" in summary
    assert "5 years" in summary
    assert "Python" in summary
    assert "Tech Corp" in summary
    assert "john@example.com" in summary
    assert "123-456-7890" in summary


def test_generate_summary_empty_data(gpt2_model, empty_resume_data):
    """Test summary generation with empty data."""
    summary = gpt2_model.generate_summary(empty_resume_data)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert summary != "Error generating summary."


def test_generate_summary_minimal_data(gpt2_model, minimal_resume_data):
    """Test summary generation with minimal data."""
    summary = gpt2_model.generate_summary(minimal_resume_data)
    
    assert isinstance(summary, str)
    assert "Jane Smith" in summary
    assert "2 years" in summary
    assert "Python" in summary
    assert "Startup Inc" in summary


def test_clean_summary(gpt2_model):
    """Test summary cleaning."""
    dirty_summary = "hi,  this is  a Test.  with extra  spaces.  "
    clean_summary = gpt2_model._clean_summary(dirty_summary)
    
    assert clean_summary == "Hi, this is a Test. With extra spaces"
    assert "  " not in clean_summary


def test_validate_summary(gpt2_model):
    """Test summary validation."""
    # Valid summary
    valid_summary = "This is a valid summary with enough words to pass the minimum length check but not too many to fail the maximum length check."
    assert gpt2_model._validate_summary(valid_summary) is True
    
    # Too short
    short_summary = "Too short."
    assert gpt2_model._validate_summary(short_summary) is False
    
    # Too long
    long_summary = " ".join(["word"] * 250)
    assert gpt2_model._validate_summary(long_summary) is False
    
    # Empty
    assert gpt2_model._validate_summary("") is False


@pytest.mark.parametrize("error_location", ["model", "tokenizer"])
def test_model_initialization_error(error_location):
    """Test error handling during model initialization."""
    with patch('src.models.gpt2_model.GPT2LMHeadModel.from_pretrained') as mock_model:
        with patch('src.models.gpt2_model.GPT2Tokenizer.from_pretrained') as mock_tokenizer:
            if error_location == "model":
                mock_model.side_effect = Exception("Model error")
            else:
                mock_tokenizer.side_effect = Exception("Tokenizer error")
                
            with pytest.raises(Exception):
                GPT2Model()


def test_format_name(gpt2_model):
    """Test name formatting."""
    assert gpt2_model._format_name("john doe") == "John Doe"
    assert gpt2_model._format_name("JANE SMITH") == "Jane Smith"
    assert gpt2_model._format_name("robert j. williams") == "Robert J. Williams"
    assert gpt2_model._format_name("") == ""


def test_clean_metrics(gpt2_model):
    """Test metrics cleaning."""
    # Test percentages
    assert gpt2_model._clean_metrics("*40*%") == "40%"
    assert gpt2_model._clean_metrics("*40.5*%") == "40.5%"
    
    # Test currency
    assert gpt2_model._clean_metrics("*$50K*") == "$50K"
    assert gpt2_model._clean_metrics("*$1.5M*") == "$1.5M"
    
    # Test numbers and ranges
    assert gpt2_model._clean_metrics("*2* times") == "2 times"
    assert gpt2_model._clean_metrics("*5*-*10*") == "5-10"
    assert gpt2_model._clean_metrics("*1* to *3*") == "1 to 3"


def test_generate_summary_error_handling(gpt2_model):
    """Test error handling in summary generation."""
    with patch.object(gpt2_model, '_clean_summary', side_effect=Exception("Test error")):
        summary = gpt2_model.generate_summary({"name": "Test User"})
        assert summary == "Error generating summary."


def test_clean_summary_patterns(gpt2_model):
    """Test various cleaning patterns in summary."""
    # Test bullet removal
    summary = "Skills: • Python • Java"
    cleaned = gpt2_model._clean_summary(summary)
    assert "•" not in cleaned
    
    # Test section header removal
    summary = "[Skills and Expertise] Python developer"
    cleaned = gpt2_model._clean_summary(summary)
    assert "[Skills and Expertise]" not in cleaned
    
    # Test self-reference cleaning
    summary = "My name is John Doe and I am a developer"
    cleaned = gpt2_model._clean_summary(summary)
    assert "My name is" not in cleaned
    assert "John Doe" in cleaned
    
    # Test transition improvement
    summary = "Recently, I worked on a project"
    cleaned = gpt2_model._clean_summary(summary)
    assert "Recently" not in cleaned
    assert any(phrase in cleaned for phrase in [
        "In my current position",
        "As part of my role",
        "In my professional journey"
    ])
