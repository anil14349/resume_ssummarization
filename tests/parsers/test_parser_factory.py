"""Tests for parser factory."""
import pytest
from unittest.mock import patch, MagicMock
from src.parsers.parser_factory import ParserFactory


@pytest.fixture
def mock_resume_file(tmp_path):
    """Create a temporary resume file."""
    resume_file = tmp_path / "test_resume.docx"
    resume_file.write_text("Test resume content")
    return str(resume_file)


def test_create_ats_parser(mock_resume_file):
    """Test ATS parser creation."""
    with patch('src.parsers.parser_factory.ATSParser') as mock_ats:
        mock_instance = MagicMock()
        mock_ats.return_value = mock_instance
        
        parser = ParserFactory.create_parser("ats", mock_resume_file)
        assert parser == mock_instance
        mock_ats.assert_called_once_with(mock_resume_file)


def test_create_industry_parser(mock_resume_file):
    """Test industry parser creation."""
    with patch('src.parsers.parser_factory.IndustryManagerParser') as mock_industry:
        mock_instance = MagicMock()
        mock_industry.return_value = mock_instance
        
        parser = ParserFactory.create_parser("industry", mock_resume_file)
        assert parser == mock_instance
        mock_industry.assert_called_once_with(mock_resume_file)


def test_create_invalid_parser(mock_resume_file):
    """Test invalid parser type handling."""
    with pytest.raises(ValueError, match="Invalid parser type"):
        ParserFactory.create_parser("invalid_parser", mock_resume_file)


def test_create_parser_no_file():
    """Test parser creation without file."""
    with pytest.raises(ValueError, match="Input file is required"):
        ParserFactory.create_parser("ats", None)


def test_create_parser_empty_file():
    """Test parser creation with empty file path."""
    with pytest.raises(ValueError, match="Input file is required"):
        ParserFactory.create_parser("ats", "")


@pytest.mark.parametrize("parser_type,parser_class", [
    ("ats", "ATSParser"),
    ("industry", "IndustryManagerParser")
])
def test_parser_type_mapping(mock_resume_file, parser_type, parser_class):
    """Test parser type to class mapping."""
    with patch(f'src.parsers.parser_factory.{parser_class}') as mock_parser:
        mock_instance = MagicMock()
        mock_parser.return_value = mock_instance
        
        parser = ParserFactory.create_parser(parser_type, mock_resume_file)
        assert parser == mock_instance
        mock_parser.assert_called_once_with(mock_resume_file)
