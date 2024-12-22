"""ATS resume parser."""
from docx import Document
import json
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ATSParser:
    def __init__(self, file_path):
        self.file_path = file_path
        logger.debug(f"Initializing ATSParser with file: {file_path}")

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
            years = (end.year - start_date.year) + 
                    (end.month - start_date.month) / 12
            total_years += max(0, years)  # Ensure non-negative
            
            logger.debug(f"Date range {start_date} - {end}: {years:.1f} years")
            
        logger.debug(f"Total years experience: {total_years:.1f}")
        return round(total_years, 1)

    def extract_company_info(self, text):
        """Extract company name and dates from text."""
        logger.debug(f"Extracting company info from: '{text}'")
        
        # Initialize result
        result = {
            'company': None,
            'dates': None
        }
        
        # Common company indicators
        company_indicators = [
            r'at\s+([^,|]+)',
            r'with\s+([^,|]+)',
            r'for\s+([^,|]+)',
            r'\|\s*([^|,]+)\s*\|'
        ]
        
        # Try to extract company name
        for pattern in company_indicators:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['company'] = match.group(1).strip()
                logger.debug(f"Found company: {result['company']}")
                break
                
        # Extract dates
        date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+20\d{2}|20\d{2}|Present|Current)'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        
        if len(dates) >= 2:
            start_date = self.parse_date(dates[0])
            end_date = self.parse_date(dates[1])
            if start_date and end_date:
                result['dates'] = (start_date, end_date)
                logger.debug(f"Found date range: {start_date} - {end_date}")
        elif len(dates) == 1:
            # Single date might be start date of current position
            start_date = self.parse_date(dates[0])
            if start_date:
                result['dates'] = (start_date, None)  # None indicates current
                logger.debug(f"Found single date (assumed current): {start_date}")
                
        return result

    def extract_role(self, text):
        """Extract role from text."""
        logger.debug(f"Extracting role from: '{text}'")
        
        # Common role patterns
        role_patterns = [
            r'^([^|]+)\|',  # Everything before first pipe
            r'^([^,]+),',   # Everything before first comma
            r'(.*?)\s+at\s+',  # Everything before " at "
            r'(.*?)\s+with\s+'  # Everything before " with "
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, text)
            if match:
                role = match.group(1).strip()
                logger.debug(f"Found role: {role}")
                return role
                
        logger.debug("No role found")
        return None

    def extract_skills(self, text):
        """Extract skills from text."""
        logger.debug(f"Extracting skills from: '{text}'")
        
        # Common HR skills
        skill_keywords = [
            'Recruitment', 'Talent Acquisition', 'Employee Relations',
            'HR Policies', 'Compliance', 'Performance Management',
            'Training', 'Employee Engagement', 'Leadership',
            'HRIS', 'Benefits Administration', 'Onboarding',
            'Compensation', 'Labor Relations', 'Diversity',
            'Workforce Planning', 'Change Management'
        ]
        
        found_skills = []
        for skill in skill_keywords:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                found_skills.append(skill)
                logger.debug(f"Found skill: {skill}")
                
        return found_skills

    def extract_achievement(self, text):
        """Extract achievement from text."""
        logger.debug(f"Extracting achievement from: '{text}'")
        
        # Skip if too short
        if len(text.split()) < 4:
            logger.debug("Text too short, skipping")
            return None
            
        # Look for metrics
        metrics_pattern = r'\d+%|\$\d+|\d+\s*million|\d+\s*k|\d+\s*employees'
        has_metrics = bool(re.search(metrics_pattern, text, re.IGNORECASE))
        
        # Look for action verbs
        action_verbs = [
            'achieved', 'improved', 'trained', 'managed', 'developed',
            'created', 'implemented', 'reduced', 'increased', 'led',
            'launched', 'established', 'coordinated', 'streamlined',
            'generated', 'saved', 'delivered', 'built', 'designed'
        ]
        has_action = any(re.search(r'\b' + verb + r'\b', text, re.IGNORECASE) 
                        for verb in action_verbs)
        
        if has_metrics or has_action:
            logger.debug(f"Found achievement: {text}")
            logger.debug(f"Has metrics: {has_metrics}, Has action verb: {has_action}")
            return text.strip()
            
        logger.debug("No achievement indicators found")
        return None

    def parse(self):
        """Parse the resume document."""
        logger.info(f"Starting to parse resume: {self.file_path}")
        
        try:
            doc = Document(self.file_path)
        except Exception as e:
            logger.error(f"Failed to open document: {e}")
            raise
            
        # Initialize data structure
        data = {
            'name': '',
            'current_role': '',
            'companies': [],
            'years_experience': 0,
            'skills': [],
            'achievements': []
        }
        
        # Track work experience dates for calculating total experience
        work_dates = []
        
        # Parse each paragraph
        for i, para in enumerate(doc.paragraphs):
            text = self.clean_text(para.text)
            if not text:
                continue
                
            logger.debug(f"Processing paragraph {i}: '{text}'")
            
            # Skip contact info
            if self.is_contact_info(text):
                logger.debug("Skipping contact info")
                continue
                
            # Extract name if not found (usually first non-empty line)
            if not data['name'] and not any(x in text.lower() for x in ['summary', 'experience', 'education']):
                data['name'] = text
                logger.debug(f"Found name: {data['name']}")
                continue
            
            # Extract company info
            company_info = self.extract_company_info(text)
            if company_info:
                if company_info['company']:
                    data['companies'].append(company_info['company'])
                if not data['current_role']:
                    data['current_role'] = self.extract_role(text)
                if company_info['dates']:
                    work_dates.append(company_info['dates'])
                continue
            
            # Extract achievement
            achievement = self.extract_achievement(text)
            if achievement:
                data['achievements'].append(achievement)
                continue
            
            # Extract skills
            skills = self.extract_skills(text)
            if skills:
                data['skills'].extend([s for s in skills if s not in data['skills']])
        
        # Calculate total years of experience
        data['years_experience'] = self.calculate_years_experience(work_dates)
        
        # Log final parsed data
        logger.info("Parsing complete. Final data:")
        logger.info(f"Name: {data['name']}")
        logger.info(f"Current Role: {data['current_role']}")
        logger.info(f"Companies: {data['companies']}")
        logger.info(f"Years Experience: {data['years_experience']}")
        logger.info(f"Skills: {data['skills']}")
        logger.info(f"Achievements: {data['achievements']}")
        
        return data

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
        extracted_data['name'] = self.extract_name(paragraphs)
        
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
                skills = self.extract_skills(text)
                for skill in skills:
                    if skill not in extracted_data['skills']:
                        extracted_data['skills'].append(skill)
            
            elif current_section == 'experience':
                if '|' in text:
                    info = self.extract_company_info(text)
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
                    achievement = self.extract_achievement(text)
                    if achievement and achievement not in extracted_data['achievements']:
                        extracted_data['achievements'].append(achievement)
                        # Extract skills from achievements
                        skills = self.extract_skills(text)
                        for skill in skills:
                            if skill not in extracted_data['skills']:
                                extracted_data['skills'].append(skill)
            
            elif current_section == 'achievements':
                achievement = self.extract_achievement(text)
                if achievement and achievement not in extracted_data['achievements']:
                    extracted_data['achievements'].append(achievement)
                    # Extract skills from achievements
                    skills = self.extract_skills(text)
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

    def extract_name(self, paragraphs):
        """Extract name from text."""
        logger.debug(f"Extracting name from paragraphs: {paragraphs}")
        
        # Common name patterns
        name_patterns = [
            r'^([A-Za-z\s]+)$',  # Simple name
            r'^([A-Za-z\s]+),\s*([A-Za-z\s]+)$'  # Name with comma
        ]
        
        for pattern in name_patterns:
            for para in paragraphs:
                match = re.search(pattern, para)
                if match:
                    name = match.group(0).strip()
                    logger.debug(f"Found name: {name}")
                    return name
                
        logger.debug("No name found")
        return ''

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
    parser = ATSParser(file_path)
    parsed_data = parser.parse_docx_to_json()
    print(json.dumps(parsed_data, indent=4))
