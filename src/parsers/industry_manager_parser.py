"""Parser for industry manager resumes."""
import logging
import re
from datetime import datetime
from typing import List, Dict, Any
from docx import Document

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class IndustryManagerParser(BaseParser):
    """Parser for industry manager resumes."""
    
    def __init__(self):
        """Initialize industry manager parser."""
        super().__init__()
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse an industry manager resume file.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Dictionary containing parsed resume data
        """
        try:
            # Read document
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
            
            # Extract information
            name = self._extract_name(text)
            current_role = self._extract_role(text)
            companies = self._extract_companies(text)
            years_experience = self._extract_years_experience(text)
            skills = self._extract_skills(text)
            achievements = self._extract_achievements(text)
            
            # Clean and validate data
            name = str(name).strip() if name else ""
            current_role = str(current_role).strip() if current_role else ""
            companies = [str(c).strip() for c in companies if c and str(c).strip()]
            years_experience = float(years_experience) if years_experience else 0.0
            skills = [str(s).strip() for s in skills if s and str(s).strip()]
            achievements = [str(a).strip() for a in achievements if a and str(a).strip()]
            
            # Build result dictionary
            result = {
                'name': name,
                'current_role': current_role,
                'companies': companies,
                'years_experience': years_experience,
                'skills': skills,
                'achievements': achievements
            }
            
            # Log the parsed data
            logger.info("Parsing complete. Final data:")
            logger.info(f"Name: {result['name']}")
            logger.info(f"Current_Role: {result['current_role']}")
            logger.info(f"Companies: {result['companies']}")
            logger.info(f"Years_Experience: {result['years_experience']}")
            logger.info(f"Skills: {result['skills']}")
            logger.info(f"Achievements: {result['achievements']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            raise
    
    def _extract_name(self, text: str) -> str:
        """Extract name from text."""
        # Look for name at the start of the document
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            # Look for capitalized words that could be a name
            words = line.strip().split()
            if len(words) >= 2 and all(w[0].isupper() for w in words if w):
                return line.strip()
        return ""
    
    def _extract_role(self, text: str) -> str:
        """Extract current role from text."""
        role_patterns = [
            r'(?i)current role:\s*([^\n]+)',
            r'(?i)position:\s*([^\n]+)',
            r'(?i)title:\s*([^\n]+)',
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Look for role in first section
        lines = text.split('\n')
        for line in lines[1:5]:  # Check lines 2-5
            if any(x in line.lower() for x in ['manager', 'director', 'lead', 'head']):
                return line.strip()
        
        return ""
    
    def _extract_companies(self, text: str) -> List[str]:
        """Extract companies from text."""
        companies = []
        
        # Look for company sections
        company_patterns = [
            r'(?i)company:\s*([^\n]+)',
            r'(?i)employer:\s*([^\n]+)',
            r'(?i)organization:\s*([^\n]+)',
            r'\b[A-Z][a-zA-Z\s&]+(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation|Company)\b'
        ]
        
        for pattern in company_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                company = match.group(1).strip() if len(match.groups()) > 0 else match.group(0)
                if company and company not in companies:
                    companies.append(company)
        
        return companies[:3]  # Return top 3 companies
    
    def _extract_years_experience(self, text: str) -> float:
        """Extract years of experience from text."""
        # Look for explicit mentions of years
        year_patterns = [
            r'(?i)(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'(?i)experience:\s*(\d+)\+?\s*(?:years?|yrs?)',
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Calculate from employment dates
        date_pattern = r'(?i)(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}'
        dates = re.findall(date_pattern, text)
        if len(dates) >= 2:
            try:
                dates = [datetime.strptime(d, '%B %Y') for d in dates]
                years = (max(dates) - min(dates)).days / 365.25
                return round(years, 1)
            except ValueError:
                pass
        
        return 0.0  # Default if no experience found
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text."""
        skills = set()
        
        # Look for skills section
        skills_section = re.search(r'(?i)skills[:\n]+(.*?)(?:\n\n|\Z)', text, re.DOTALL)
        if skills_section:
            # Split by common delimiters
            skill_text = skills_section.group(1)
            skill_list = re.split(r'[,;•|\n]', skill_text)
            
            # Clean and add skills
            for skill in skill_list:
                skill = skill.strip()
                if skill and len(skill) > 2:  # Ignore very short skills
                    skills.add(skill)
        
        # Look for key technical terms
        technical_terms = [
            'management', 'leadership', 'strategy', 'operations',
            'business development', 'sales', 'marketing', 'finance',
            'analytics', 'project management', 'team building'
        ]
        
        for term in technical_terms:
            if term.lower() in text.lower():
                skills.add(term)
        
        return list(skills)
    
    def _extract_achievements(self, text: str) -> List[str]:
        """Extract achievements from text."""
        achievements = []
        
        # Look for achievements section
        achievement_section = re.search(r'(?i)(?:achievements?|accomplishments?)[:\n]+(.*?)(?:\n\n|\Z)', text, re.DOTALL)
        if achievement_section:
            # Split by bullet points or newlines
            achievement_text = achievement_section.group(1)
            achievement_list = re.split(r'[•\n]', achievement_text)
            
            # Clean and filter achievements
            for achievement in achievement_list:
                achievement = achievement.strip()
                if achievement and len(achievement) > 20:  # Ignore short lines
                    if any(x in achievement.lower() for x in ['increased', 'decreased', 'improved', 'led', 'managed', 'developed']):
                        achievements.append(achievement)
        
        # Look for achievements in experience section
        experience_section = re.search(r'(?i)experience[:\n]+(.*?)(?:\n\n|\Z)', text, re.DOTALL)
        if experience_section:
            lines = experience_section.group(1).split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 20:
                    if any(x in line.lower() for x in ['increased', 'decreased', 'improved', 'led', 'managed', 'developed']):
                        if any(x in line for x in ['%', '$', '+', 'million', 'billion']):
                            achievements.append(line)
        
        return achievements[:5]  # Return top 5 achievements

if __name__ == "__main__":
    import json
    file_path = "src/templates/Industry manager resume.docx"
    parser = IndustryManagerParser()
    result = parser.parse(file_path)
    print(json.dumps(result, indent=4))
