"""Tests for industry manager parser."""
import pytest
from src.parsers.industry_manager_parser import IndustryManagerParser
from docx import Document
import os

@pytest.fixture
def sample_resume_path(tmp_path):
    """Create a sample resume for testing."""
    doc = Document()
    
    # Add contact info in first line
    doc.add_paragraph("123 Main St, City, State 12345 | (555) 123-4567 | j.smith@example.com | www.linkedin.com/in/jsmith")
    
    # Add Profile section
    doc.add_paragraph("Profile")
    doc.add_paragraph("Experienced manager with strong leadership skills and proven track record in team building and process improvement.")
    
    # Add Experience section
    doc.add_paragraph("Experience")
    doc.add_paragraph("Restaurant Manager | Contoso Grill | September 2020 - Present")
    doc.add_paragraph("• Increased revenue by 25% through strategic menu optimization")
    doc.add_paragraph("• Led a team of 15 staff members")
    doc.add_paragraph("Assistant Manager | Tech Cafe | June 2015 - August 2020")
    doc.add_paragraph("• Improved customer satisfaction by 30%")
    
    # Add Education section
    doc.add_paragraph("Education")
    doc.add_paragraph("B.S. in Business Management | State University | 2015")
    
    # Add Skills & Abilities section
    doc.add_paragraph("Skills & Abilities")
    doc.add_paragraph("Leadership, Team Building, Customer Service, Operations Management, Food Service")
    
    # Add Activities section
    doc.add_paragraph("Activities and Interests")
    doc.add_paragraph("Volunteer work, cooking, travel")
    
    # Save the document
    file_path = tmp_path / "test_resume.docx"
    doc.save(file_path)
    return str(file_path)

@pytest.fixture
def parser():
    """Create parser instance."""
    return IndustryManagerParser()

def test_parse_resume(parser, sample_resume_path):
    """Test parsing a complete resume."""
    result = parser.parse(sample_resume_path)
    
    assert isinstance(result, dict)
    assert result['name'] == "J. Smith"
    assert result['current_role'] == "Restaurant Manager"
    assert result['companies'] == ["Contoso Grill", "Tech Cafe"]
    assert result['years_experience'] == 9.0  # Current year - 2015
    
    # Skills can come in different variations, check for presence rather than exact match
    skills = set(result['skills'][:5])
    assert "Customer Service" in skills
    assert any(s for s in skills if "Lead" in s)  # Could be Leader or Leadership
    assert "Operations Management" in skills or "Management" in skills
    
    assert len(result['achievements']) <= 3
    assert any("revenue by 25%" in a for a in result['achievements'])
    assert any("team of 15" in a for a in result['achievements'])
    
    # Test contact information
    assert result['contact_info']['email'] == "j.smith@example.com"
    assert result['contact_info']['phone'] == "555-123-4567"

def test_extract_name(parser):
    """Test name extraction."""
    text = "123 Main St | (555) 123-4567 | j.smith@example.com | linkedin"
    assert parser._extract_name(text) == "J. Smith"

def test_extract_role(parser):
    """Test role extraction."""
    text = "Experience\nRestaurant Manager | Contoso Grill | September 2020 - Present"
    assert parser._extract_role(text) == "Restaurant Manager"

def test_extract_companies(parser):
    """Test company extraction."""
    text = "Experience\nManager | Contoso Grill | September 2020 - Present\nLead | Tech Cafe | June 2015 - August 2020"
    companies = parser._extract_companies(text)
    assert companies == ["Contoso Grill", "Tech Cafe"]

def test_extract_years_experience(parser):
    """Test years of experience extraction."""
    text = "Experience\nManager | Company | September 2020 - Present\nLead | Other Co | June 2015 - August 2020"
    assert parser._extract_years_experience(text) == 9.0  # Current year - 2015

def test_extract_skills(parser):
    """Test skills extraction."""
    text = """Profile
A skilled leader with customer service focus and team building experience.

Skills & Abilities
Leader, Customer Service, Team Building, Operations Management

Activities and Interests
Other activities"""
    
    skills = parser._extract_skills(text)
    # Check for presence of key skills, allowing for variations
    assert any(s for s in skills if "Lead" in s)  # Could be Leader or Leadership
    assert "Customer Service" in skills
    assert "Team Building" in skills

def test_extract_achievements(parser):
    """Test achievements extraction."""
    text = """Experience
    Manager | Company | Date - Present
    • Increased revenue by 25% through optimization
    • Led a team of 15 staff members
    • Improved efficiency by 40%"""
    achievements = parser._extract_achievements(text)
    assert len(achievements) <= 3
    assert any("revenue by 25%" in a for a in achievements)
    assert any("team of 15" in a for a in achievements)

def test_extract_contact_info(parser):
    """Test contact information extraction."""
    text = "123 Main St | (555) 123-4567 | j.smith@example.com | www.linkedin.com/in/jsmith"
    contact_info = parser._extract_contact_info(text)
    assert contact_info['email'] == "j.smith@example.com"
    assert contact_info['phone'] == "555-123-4567"

def test_parse_empty_resume(parser, tmp_path):
    """Test parsing an empty resume."""
    # Create empty document
    doc = Document()
    file_path = tmp_path / "empty_resume.docx"
    doc.save(file_path)
    
    result = parser.parse(str(file_path))
    assert isinstance(result, dict)
    assert result['name'] == ""
    assert result['current_role'] == ""
    assert result['companies'] == []
    assert result['years_experience'] == 0.0
    assert result['skills'] == []
    assert result['achievements'] == []
    assert result['contact_info'] == {'email': '', 'phone': ''}
