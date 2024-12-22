"""ATS resume parser.
this code uses ATS classic HR resume.docx file as template format to extract data, which is under templates folder
"""
import logging
import re
from datetime import datetime
from typing import List, Dict, Any
from docx import Document

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class ATSParser(BaseParser):
    """Parser for ATS-formatted resumes."""
    
    def __init__(self):
        """Initialize ATS parser."""
        super().__init__()

    def clean_text(self, text):
        """Clean and normalize text."""
        cleaned = ' '.join(text.strip().split())
        logger.debug(f"Cleaned text: '{text}' -> '{cleaned}'")
        return cleaned

    def parse_date(self, text):
        """Parse date from text in various formats."""
        logger.debug(f"Parsing date from: '{text}'")
        
        # Remove any non-alphanumeric characters from the end
        text = text.strip().rstrip('.')
        
        # Handle 'Present' or 'Current'
        if text.lower() in ['present', 'current']:
            logger.debug("Found 'present/current', using current date")
            return datetime.now()
        
        # Handle 20XX format
        if re.match(r'20XX', text, re.IGNORECASE):
            logger.debug("Found '20XX' format, using 2021")
            return datetime(2021, 1, 1)  # Assume recent
        
        # Common date formats
        date_patterns = [
            (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+20\d{2}', '%B %Y'),
            (r'20\d{2}', '%Y')
        ]
        
        for pattern, date_format in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(0)
                    # Standardize month abbreviations
                    date_str = re.sub(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*',
                                    lambda m: m.group(0)[:3], date_str, flags=re.IGNORECASE)
                    date_str = re.sub(r'[,\s]+', ' ', date_str).strip()
                    parsed_date = datetime.strptime(date_str, date_format)
                    logger.debug(f"Successfully parsed date: {date_str} -> {parsed_date}")
                    return parsed_date
                except ValueError as e:
                    logger.debug(f"Failed to parse date with format {date_format}: {e}")
                    continue
        
        logger.debug("Could not parse date")
        return None

    def calculate_years_experience(self, dates):
        """Calculate total years of experience from date ranges."""
        if not dates:
            logger.debug("No dates provided for experience calculation")
            return 0
            
        logger.debug(f"Calculating experience from dates: {dates}")
        total_years = 0
        current_date = datetime.now()
        
        for start_date, end_date in dates:
            if not start_date:
                logger.debug(f"Skipping date range due to missing start date: {start_date} - {end_date}")
                continue
                
            # Use current date if end_date is None (current position)
            end = end_date if end_date else current_date
            
            # Calculate years
            years = (end.year - start_date.year) + (end.month - start_date.month) / 12
            total_years += max(0, years)  # Ensure non-negative
            
            logger.debug(f"Date range {start_date} - {end}: {years:.1f} years")
            
        logger.debug(f"Total years experience: {total_years:.1f}")
        return round(total_years, 1)

    def _extract_company(self, text: str) -> str:
        """Extract company name from text."""
        # Look for patterns like "Company Name | Role" or "Role | Company Name"
        company_patterns = [
            r'(?:at|with|for)\s+([A-Z][A-Za-z\s]+(?:Inc|LLC|Ltd|Corp|Company|Healthcare|Solutions))',
            r'\|\s*([A-Z][A-Za-z\s]+(?:Inc|LLC|Ltd|Corp|Company|Healthcare|Solutions))',
            r'([A-Z][A-Za-z\s]+(?:Inc|LLC|Ltd|Corp|Company|Healthcare|Solutions))\s*\|'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the longest match as it's likely the most complete
                return max(matches, key=len).strip()
        
        # If no match found, look for capitalized words that might be company names
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Check if next word is also capitalized (likely part of company name)
                if i < len(words) - 1 and words[i + 1][0].isupper():
                    return f"{word} {words[i + 1]}".strip()
                return word.strip()
        
        return "Unknown Company"

    def _extract_achievements(self, text: str) -> List[str]:
        """Extract achievements from text."""
        achievements = []
        
        # Look for bullet points or numbered achievements
        achievement_patterns = [
            r'[•\-\*]\s*(.*?)(?=(?:[•\-\*]|\n|$))',
            r'\d+\.\s*(.*?)(?=(?:\d+\.|\n|$))',
            r'(?:^|\n)(?!.*?(?:education|skills|experience|profile):)([A-Z][^.!?]*?(?:increased|decreased|reduced|improved|developed|implemented|led|managed|created|designed|launched|achieved|won|earned|saved|generated)[^.!?]*[.!?])'
        ]
        
        # Look for sentences with metrics or key achievements
        metric_patterns = [
            r'(?:increased|improved|reduced|decreased|saved|generated|achieved)\s+[^.!?]*?(?:\d+%|\$\d+|\d+\s*(?:percent|million|billion|thousand))[^.!?]*[.!?]',
            r'(?:led|managed|developed|implemented|created|designed|launched)\s+[^.!?]*?(?:team|project|initiative|program|system)[^.!?]*[.!?]',
            r'(?:successfully|effectively)\s+[^.!?]*?(?:improved|increased|reduced|developed|implemented)[^.!?]*[.!?]'
        ]
        
        # Process achievement patterns
        for pattern in achievement_patterns + metric_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                achievement = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                
                # Clean up achievement text
                achievement = re.sub(r'\s+', ' ', achievement)  # Replace multiple spaces
                achievement = achievement.strip('.')  # Remove trailing period
                
                # Only keep substantial and relevant achievements
                if (
                    achievement 
                    and len(achievement.split()) >= 4  # Must be substantial
                    and not any(x in achievement.lower() for x in ['education', 'gpa', 'honor society', 'university'])  # Skip education
                    and not any(x.lower() == achievement.lower() for x in achievements)  # Avoid duplicates
                ):
                    # Clean up and format
                    achievement = achievement[0].upper() + achievement[1:]  # Capitalize first letter
                    if not achievement.endswith(('.', '!', '?')):
                        achievement += '.'
                    achievements.append(achievement)
        
        # Sort achievements by length and quality
        achievements.sort(key=lambda x: (
            # Prioritize achievements with metrics
            -len(re.findall(r'\d+%|\$\d+|\d+\s*(?:percent|million|billion|thousand)', x)),
            # Then by presence of key verbs
            -len(re.findall(r'increased|improved|reduced|developed|implemented|led|managed', x.lower())),
            # Then by length (prefer longer, more detailed achievements)
            -len(x)
        ))
        
        # Return top 3 achievements
        return achievements[:3]

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text."""
        skills = set()
        
        # Common HR skills
        hr_skills = [
            'recruitment', 'hiring', 'onboarding', 'training', 'development',
            'performance management', 'employee relations', 'benefits administration',
            'compensation', 'payroll', 'HRIS', 'compliance', 'policy development',
            'talent acquisition', 'succession planning', 'workforce planning',
            'employee engagement', 'diversity', 'inclusion', 'labor relations',
            'conflict resolution', 'leadership development', 'organizational development',
            'change management', 'project management', 'data analytics', 'reporting',
            'budgeting', 'vendor management', 'benefits', 'employee satisfaction',
            'talent management', 'HR strategy', 'employee retention', 'HR policies',
            'HR programs', 'employee communications', 'employee experience',
            'performance evaluation', 'team building', 'coaching', 'mentoring',
            'employee advocacy', 'workplace culture', 'HR metrics', 'HR analytics',
            'employee surveys', 'HRMS', 'ATS', 'HRIS implementation'
        ]
        
        # Look for skills in text
        text_lower = text.lower()
        for skill in hr_skills:
            if skill.lower() in text_lower:
                skills.add(skill)
        
        # Look for additional skills
        skill_patterns = [
            r'(?:proficient|skilled|expertise|experienced)\s+(?:in|with)?\s+([^.]*)',
            r'(?:skills|abilities):\s*([^.]*)',
            r'(?:^|\n)(?:•|\*|\-|\d+\.)\s*([A-Za-z\s]+(?:management|planning|development|analysis|implementation|design|coordination))'
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill_text = match.group(1).strip()
                # Split into individual skills
                for skill in skill_text.split(','):
                    skill = skill.strip()
                    if skill and len(skill.split()) <= 3:  # Keep skills concise
                        skills.add(skill)
        
        return list(skills)

    def _extract_years_experience(self, text: str) -> float:
        """Extract years of experience from text."""
        # Look for explicit mentions of years
        year_patterns = [
            r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'(?:experience|work(?:ing)?|professional)\s+(?:of|for)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)',
            r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:in|as)\s+(?:HR|Human\s+Resources|the\s+field)'
        ]
        
        for pattern in year_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = float(match.group(1))
                    if years > 0 and years < 50:  # Sanity check
                        return years
                except ValueError:
                    continue
        
        # If no explicit mention, try to calculate from work history
        date_pattern = r'(?:19|20)\d{2}'
        dates = re.findall(date_pattern, text)
        if dates:
            dates = [int(d) for d in dates]
            years = max(dates) - min(dates)
            if years > 0 and years < 50:
                return float(years)
        
        # Default to a reasonable value if no clear indication
        return 5.0

    def _extract_companies(self, text: str) -> List[str]:
        """Extract companies from text."""
        companies = []
        
        # Look for company names
        company_patterns = [
            r'(?:at|with|for)\s+([A-Z][A-Za-z\s]+(?:Inc|LLC|Ltd|Corp|Company|Healthcare|Solutions))',
            r'\|\s*([A-Z][A-Za-z\s]+(?:Inc|LLC|Ltd|Corp|Company|Healthcare|Solutions))',
            r'([A-Z][A-Za-z\s]+(?:Inc|LLC|Ltd|Corp|Company|Healthcare|Solutions))\s*\|'
        ]
        
        for pattern in company_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                company = match.group(1).strip()
                if company and not any(c.lower() == company.lower() for c in companies):
                    companies.append(company)
        
        # If no matches found, look for company-like names
        if not companies:
            # Split text into lines and look for company-like names
            lines = text.split('\n')
            for line in lines:
                # Look for lines that might contain company names
                if '|' in line:  # Common format in resumes
                    parts = line.split('|')
                    for part in parts:
                        part = part.strip()
                        if part and part[0].isupper() and 'experience' not in part.lower():
                            companies.append(part)
                            
        # If still no companies found, look for capitalized phrases
        if not companies:
            # Look for consecutive capitalized words
            matches = re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
            for match in matches:
                company = match.group(1).strip()
                if company and not any(c.lower() == company.lower() for c in companies):
                    companies.append(company)
        
        # If no companies found at all, use a default
        if not companies:
            companies = ["Unknown Company"]
            
        return companies[:3]  # Return at most 3 companies

    def _extract_name(self, text: str) -> str:
        """Extract name from text."""
        # Look for name at the start of the document
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            # Skip empty lines and common headers
            if not line.strip() or any(header in line.lower() for header in ['resume', 'cv', 'curriculum']):
                continue
                
            # Look for a name (2-3 words, each capitalized)
            words = line.strip().split()
            if 2 <= len(words) <= 3 and all(word[0].isupper() for word in words):
                return ' '.join(words)
                
            # Look for a name before contact info
            if any(info in line.lower() for info in ['@', 'email', 'phone', 'address']):
                before_contact = line.split('@')[0] if '@' in line else line.split('•')[0]
                words = before_contact.strip().split()
                if 2 <= len(words) <= 3 and all(word[0].isupper() for word in words):
                    return ' '.join(words)
        
        return ""

    def _extract_role(self, text: str) -> str:
        """Extract current role from text."""
        # Look for role in professional summary
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for common role indicators
            role_indicators = [
                'years of experience',
                'professional with',
                'specialist in',
                'expert in',
                'generalist',
                'manager',
                'director'
            ]
            
            if any(indicator in line.lower() for indicator in role_indicators):
                # Extract the role part (usually before 'with' or similar)
                for splitter in ['with', 'having', 'possessing']:
                    if splitter in line.lower():
                        role = line.split(splitter)[0].strip()
                        if 3 <= len(role.split()) <= 10:  # Reasonable length for a role
                            return role
                
                # If no splitter found but line looks like a role, return it
                if 3 <= len(line.split()) <= 15:
                    return line
        
        # Look for role in experience section
        experience_markers = ['experience', 'employment', 'work history']
        in_experience = False
        for line in lines:
            line = line.strip()
            
            # Check if we're in experience section
            if any(marker in line.lower() for marker in experience_markers):
                in_experience = True
                continue
            
            if in_experience and line:
                # Look for job titles (usually 2-5 words, capitalized)
                words = line.split()
                if 2 <= len(words) <= 5 and any(word[0].isupper() for word in words):
                    return line
        
        return ""

    def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse an ATS resume file.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Dictionary containing parsed resume data
        """
        try:
            # Parse the document
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

    def is_contact_info(self, text):
        """Check if text contains contact information."""
        patterns = [
            r'\d{3}[-.]?\d{3}[-.]?\d{4}',  # Phone number
            r'[^@]+@[^@]+\.[^@]+',          # Email
            r'\b\d+\s+[A-Za-z\s]+,\s+[A-Za-z\s]+,\s+[A-Za-z\s]+\s+\d+\b'  # Address
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def parse_docx_to_json(self):
        """Parse a .docx file to extract structured data into the specified JSON format."""
        
        extracted_data = {
            'name': '',
            'current_role': '',
            'years_experience': 0,
            'companies': [],
            'achievements': [],
            'skills': [],
            'education': [],
            'recognition': ''
        }

        document = Document(self.file_path)
        
        # First pass: Extract name and role
        paragraphs = [self.clean_text(para.text) for para in document.paragraphs if self.clean_text(para.text)]
        
        # Extract name
        extracted_data['name'] = self._extract_name(paragraphs)
        
        work_dates = []
        current_section = ''
        in_experience_details = False
        profile_text = ''
        
        for para in document.paragraphs:
            text = self.clean_text(para.text)
            if not text:
                continue

            lower_text = text.lower()

            # Section detection
            if any(section in lower_text for section in ['experience', 'employment', 'work history']) and len(text) < 30:
                current_section = 'experience'
                in_experience_details = False
                continue
            elif any(section in lower_text for section in ['achievement', 'accomplishment']) and len(text) < 30:
                current_section = 'achievements'
                continue
            elif any(word in lower_text for word in ['skills', 'proficiencies', 'expertise']) and len(text) < 30:
                current_section = 'skills'
                continue
            elif 'education' in lower_text and len(text) < 30:
                current_section = 'education'
                continue
            elif any(section in lower_text for section in ['recognition', 'award', 'honor']) and len(text) < 30:
                current_section = 'recognition'
                continue
            elif any(word in lower_text for word in ['profile', 'summary', 'objective']) and len(text) < 30:
                current_section = 'profile'
                continue

            # Process sections
            if current_section == 'profile':
                profile_text += ' ' + text
                # Extract skills from profile
                skills = self._extract_skills(text)
                for skill in skills:
                    if skill not in extracted_data['skills']:
                        extracted_data['skills'].append(skill)
            
            elif current_section == 'experience':
                if '|' in text:
                    info = self._extract_company_info(text)
                    if info:
                        if info['company'] and info['company'] not in extracted_data['companies']:
                            extracted_data['companies'].append(info['company'])
                        if info['dates']:
                            work_dates.append((info['dates'][0], info['dates'][-1]))
                        # Extract role from the first part
                        parts = text.split('|')
                        if parts and not extracted_data['current_role']:
                            role = self.clean_text(parts[0])
                            extracted_data['current_role'] = role
                    in_experience_details = True
                elif in_experience_details:
                    achievement = self._extract_achievement(text)
                    if achievement and achievement not in extracted_data['achievements']:
                        extracted_data['achievements'].append(achievement)
                        # Extract skills from achievements
                        skills = self._extract_skills(text)
                        for skill in skills:
                            if skill not in extracted_data['skills']:
                                extracted_data['skills'].append(skill)
            
            elif current_section == 'achievements':
                achievement = self._extract_achievement(text)
                if achievement and achievement not in extracted_data['achievements']:
                    extracted_data['achievements'].append(achievement)
                    # Extract skills from achievements
                    skills = self._extract_skills(text)
                    for skill in skills:
                        if skill not in extracted_data['skills']:
                            extracted_data['skills'].append(skill)
            
            elif current_section == 'education':
                if self.is_education_related(text):
                    if '|' in text:
                        parts = [p.strip() for p in text.split('|')]
                        education = next((p for p in parts if self.is_education_related(p)), None)
                        if education and education not in extracted_data['education']:
                            extracted_data['education'].append(education)
                    else:
                        education = self.clean_text(text)
                        if education and education not in extracted_data['education']:
                            extracted_data['education'].append(education)
            
            elif current_section == 'recognition':
                if not any(word in text.lower() for word in ['section', 'recognition', 'awards']):
                    if text not in extracted_data['recognition']:
                        extracted_data['recognition'] = text if not extracted_data['recognition'] else extracted_data['recognition'] + '; ' + text

        # Calculate total years of experience from work dates
        if work_dates:
            extracted_data['years_experience'] = self.calculate_years_experience(work_dates)

        return extracted_data

    def is_education_related(self, text):
        """Check if text is related to education."""
        patterns = [
            r'\b(Bachelor|Master|Doctor|PhD|Degree|Diploma|Certificate)\b',
            r'\b(University|College|School)\b'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

# Example usage
if __name__ == "__main__":
    file_path = 'src/templates/ATS classic HR resume.docx'
    parser = ATSParser()
    parser.file_path = file_path
    parsed_data = parser.parse_docx_to_json()
    print(json.dumps(parsed_data, indent=4))
