"""Tests for the main generate_summary script."""
import pytest
from unittest.mock import patch, MagicMock
import os
from src.generate_summary import main, parse_arguments


@pytest.fixture
def mock_resume_file(tmp_path):
    """Create a temporary resume file."""
    resume_file = tmp_path / "test_resume.docx"
    resume_file.write_text("Test resume content")
    return str(resume_file)


@pytest.fixture
def mock_parser():
    """Mock resume parser."""
    parser = MagicMock()
    parser.parse.return_value = {
        'name': 'Test User',
        'years_experience': 5,
        'skills': ['Python', 'Testing'],
        'companies': ['Test Corp'],
        'achievements': ['Achieved something great'],
        'contact_info': {'email': 'test@example.com'}
    }
    return parser


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = MagicMock()
    model.generate_summary.return_value = "Test summary"
    return model


def test_parse_arguments():
    """Test command line argument parsing."""
    test_args = ["test_resume.docx", "--model", "t5", "--parser", "industry", "--debug"]
    with patch('sys.argv', ['generate_summary.py'] + test_args):
        args = parse_arguments()
        
        assert args.input_file == "test_resume.docx"
        assert args.model == "t5"
        assert args.parser == "industry"
        assert args.debug is True


def test_main_success(mock_resume_file, mock_parser, mock_model):
    """Test successful summary generation."""
    with patch('src.generate_summary.ParserFactory.create_parser', return_value=mock_parser):
        with patch('src.generate_summary.create_model', return_value=mock_model):
            summary = main(mock_resume_file, "gpt2", "ats", True)
            
            assert summary == "Test summary"
            mock_parser.parse.assert_called_once()
            mock_model.generate_summary.assert_called_once()


def test_main_file_not_found():
    """Test handling of non-existent input file."""
    with pytest.raises(FileNotFoundError):
        main("nonexistent_file.docx")


def test_main_parser_error(mock_resume_file, mock_parser):
    """Test handling of parser errors."""
    mock_parser.parse.return_value = None
    with patch('src.generate_summary.ParserFactory.create_parser', return_value=mock_parser):
        with pytest.raises(ValueError, match="Failed to parse resume data"):
            main(mock_resume_file)


def test_main_model_error(mock_resume_file, mock_parser):
    """Test handling of model creation errors."""
    with patch('src.generate_summary.ParserFactory.create_parser', return_value=mock_parser):
        with patch('src.generate_summary.create_model', side_effect=Exception("Model error")):
            with pytest.raises(Exception, match="Model error"):
                main(mock_resume_file)


@pytest.mark.parametrize("model_type", ["gpt2", "t5", "bart"])
def test_main_different_models(mock_resume_file, mock_parser, mock_model, model_type):
    """Test summary generation with different models."""
    with patch('src.generate_summary.ParserFactory.create_parser', return_value=mock_parser):
        with patch('src.generate_summary.create_model', return_value=mock_model):
            summary = main(mock_resume_file, model_type)
            assert summary == "Test summary"


@pytest.mark.parametrize("parser_type", ["ats", "industry"])
def test_main_different_parsers(mock_resume_file, mock_parser, mock_model, parser_type):
    """Test summary generation with different parsers."""
    with patch('src.generate_summary.ParserFactory.create_parser', return_value=mock_parser):
        with patch('src.generate_summary.create_model', return_value=mock_model):
            summary = main(mock_resume_file, "gpt2", parser_type)
            assert summary == "Test summary"
