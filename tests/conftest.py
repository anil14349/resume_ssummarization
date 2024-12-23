"""Test fixtures for resume summary generator."""
import pytest


@pytest.fixture
def sample_resume_data():
    """Sample resume data for testing."""
    return {
        'name': 'John Doe',
        'years_experience': 5,
        'skills': ['Python', 'Machine Learning', 'Data Science', 'AWS', 'Docker'],
        'companies': ['Tech Corp', 'Data Inc'],
        'achievements': ['Increased system performance by 40%', 'Led a team of 10 engineers'],
        'contact_info': {
            'email': 'john@example.com',
            'phone': '+1-234-567-8900'
        }
    }


@pytest.fixture
def empty_resume_data():
    """Empty resume data for testing edge cases."""
    return {
        'name': '',
        'years_experience': 0,
        'skills': [],
        'companies': [],
        'achievements': [],
        'contact_info': {}
    }


@pytest.fixture
def minimal_resume_data():
    """Minimal resume data for testing."""
    return {
        'name': 'Jane Smith',
        'years_experience': 2,
        'skills': ['Python'],
        'companies': ['Startup Inc'],
        'achievements': [],
        'contact_info': {'email': 'jane@example.com'}
    }
