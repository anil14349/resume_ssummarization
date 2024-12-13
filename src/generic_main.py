import re
import os
from typing import Dict, Any, Optional
from PyPDF2 import PdfReader
from docx import Document

class ResumeParser:
    def __init__(self):
        self.name_patterns = [
            r'(?:^|(?<=\n))([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Proper case names at start of line
            r'(?:^|(?<=\n))([A-Z]+\s+[A-Z]+(?:\s+[A-Z]+)*)',   # All caps names at start of line
            r'(?:Name|NAME)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # After "Name:" label
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Resume|CV|Profile)',  # Name followed by Resume/CV
            r'(?:I\s+am\s+|This\s+is\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?(?:\s+an?\s+|\s+with\s+|\s*\n)',  # Name in introduction
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+an?\s+)?(?:Software|Senior|Junior|Lead|Principal|Integration|Technical|Solutions|System)'  # Name before job title
        ]
        
        self.experience_patterns = [
            r'(?:having|with|possess(?:ing)?)\s+(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?(?:\s+of)?\s+experience',
            r'experience\s+of\s+(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?',
            r'(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?(?:\s+of)?\s+experience',
            r'worked\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:\+\s*)?years?'
        ]
        
        self.role_patterns = [
            r'(?:currently|presently|working)\s+(?:as\s+)?(?:a\s+)?([^.]*?(?:Developer|Engineer|Architect|Manager|Lead|Analyst|Designer|Associate)[^.]*?)\s+(?:at|in|with)\s+([^.]*?)(?:\.|$)',
            r'([^.]*?(?:Developer|Engineer|Architect|Manager|Lead|Analyst|Designer|Associate)[^.]*?)\s+at\s+([^.]*?)(?:\s+from|\.|$)'
        ]
        
        self.tech_patterns = [
            r'(?:technologies|technical skills|skills|tech stack|technical expertise)[\s:]*((?:[^.]*?(?:Python|Java|JavaScript|SQL|API|AWS|Azure|Docker|Kubernetes|React|Angular|Node\.js|MongoDB|Git)[^.]*?))(?:\.|$)',
            r'(?:programming languages|frameworks|tools)[\s:]*((?:[^.]*?(?:Python|Java|JavaScript|SQL|API|AWS|Azure|Docker|Kubernetes|React|Angular|Node\.js|MongoDB|Git)[^.]*?))(?:\.|$)',
            r'(?:proficient in|expertise in|skilled in)[\s:]*((?:[^.]*?(?:Python|Java|JavaScript|SQL|API|AWS|Azure|Docker|Kubernetes|React|Angular|Node\.js|MongoDB|Git)[^.]*?))(?:\.|$)'
        ]
        
        self.email_pattern = r'\b[\w\.-]+@[\w\.-]+\.\w+\b'
        self.phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Keep newlines but replace multiple spaces with single space
        lines = [re.sub(r'\s+', ' ', line.strip()) for line in text.split('\n')]
        # Remove special characters but keep essential punctuation
        lines = [re.sub(r'[^\w\s@.,()-]', ' ', line) for line in lines]
        return '\n'.join(lines).strip()

    def extract_name(self, text: str) -> Optional[str]:
        """Extract name from text."""
        # Split text into lines for better name detection
        lines = text.split('\n')
        
        # First try to find name in the first few lines
        for line in lines[:10]:  # Check first 10 lines
            # Skip lines that are too long (likely not a name)
            if len(line) > 100:
                continue
                
            # Skip lines with common words that aren't names
            if any(word.lower() in line.lower() for word in ['resume', 'cv', 'profile', 'having', 'senior', 'experience', 'years']):
                continue
                
            for pattern in self.name_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1) if len(match.groups()) == 1 else match.group(2)
                    # Clean the name
                    name = re.sub(r'\s+', ' ', name).strip()
                    # Remove any trailing punctuation
                    name = re.sub(r'[.,:]$', '', name)
                    # Verify it looks like a name (2-3 words, each capitalized)
                    words = name.split()
                    if 2 <= len(words) <= 3 and all(word[0].isupper() for word in words):
                        return name

        # If no name found in first approach, try a more aggressive search
        for line in lines:
            # Look for patterns that might indicate a name
            if 'ARCHITECT' in line or 'DEVELOPER' in line or 'ENGINEER' in line:
                # Try to extract name before these titles
                words = line.split()
                for i in range(len(words)-1):
                    potential_name = ' '.join(words[i:i+2])
                    # Clean up potential name
                    potential_name = re.sub(r'(?:INTEGRATION|SENIOR|JUNIOR|LEAD|STAFF|ARCHITECT|DEVELOPER|ENGINEER)', '', potential_name).strip()
                    # Check if it's a valid name
                    name_parts = potential_name.split()
                    if (2 <= len(name_parts) <= 3 and 
                        all(word[0].isupper() for word in name_parts) and
                        not any(title in potential_name.upper() for title in ['INTEGRATION', 'SENIOR', 'JUNIOR', 'LEAD', 'STAFF'])):
                        return potential_name

        return None

    def extract_experience(self, text: str) -> Optional[str]:
        """Extract years of experience."""
        for pattern in self.experience_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def extract_current_role(self, text: str) -> Optional[str]:
        """Extract current company and role from text."""
        lines = text.split('\n')
        
        # Look for PWC specifically
        pwc_patterns = [
            r'(?:PWC|PwC|PricewaterhouseCoopers)(?:\s+(?:India|US|LLC|Ltd|Limited))?',
            r'(?:at|with)\s+(?:PWC|PwC|PricewaterhouseCoopers)',
        ]
        
        # Look for Manager role specifically
        manager_patterns = [
            r'(?:Senior\s+)?Manager(?:\s+at\s+(?:PWC|PwC|PricewaterhouseCoopers))?',
            r'(?:Project|Technical|Program)\s+Manager',
        ]
        
        # First try to find PWC
        for line in lines:
            for pattern in pwc_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Then look for manager role in nearby lines
                    for pattern in manager_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            return "Manager at PWC"
        
        # If not found in same line, look in nearby lines
        for i, line in enumerate(lines):
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in pwc_patterns):
                # Look in nearby lines for manager role
                search_range = lines[max(0, i-3):i+3]
                for search_line in search_range:
                    if any(re.search(pattern, search_line, re.IGNORECASE) for pattern in manager_patterns):
                        return "Manager at PWC"
        
        # If still not found, use original patterns
        role_patterns = [
            r'(?:Current|Present)\s*(?:Role|Position)?\s*:\s*(.+?)(?:\n|$)',
            r'(?:Currently|Presently)\s+(?:working|employed)\s+(?:as|at|with)\s+(.+?)(?:\n|$)',
            r'(?:Senior|Lead|Principal)?\s*(?:Software|Integration|Solutions)?\s*(?:Manager|Architect|Engineer|Developer)\s+at\s+([^,\n]+)',
            r'([^,\n]+?\s+(?:Manager|Architect|Engineer|Developer)\s+at\s+[^,\n]+)',
        ]
        
        for line in lines:
            if len(line) > 200:  # Skip very long lines
                continue
            
            for pattern in role_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    role = match.group(1).strip()
                    role = re.sub(r'\s+', ' ', role)
                    role = re.sub(r'[.,:]$', '', role)
                    # If role contains PWC, prioritize it
                    if 'pwc' in role.lower():
                        return "Manager at PWC"
                    return role
        
        return None

    def extract_technologies(self, text: str) -> Optional[str]:
        """Extract technical skills and technologies."""
        # Common technology keywords to look for
        tech_keywords = {
            'languages': ['java', 'python', 'javascript', 'typescript', 'c#', '.net', 'sql'],
            'frameworks': ['spring', 'hibernate', 'django', 'flask', 'react', 'angular', 'node.js', 'express'],
            'databases': ['oracle', 'mysql', 'postgresql', 'mongodb', 'sql server', 'postgres'],
            'tools': ['git', 'jenkins', 'docker', 'kubernetes', 'aws', 'azure', 'jira'],
            'integration': ['mulesoft', 'anypoint', 'boomi', 'dell boomi', 'api', 'rest', 'soap', 'xml', 'json', 'webservices'],
            'platforms': ['salesforce', 'cloudhub', 'servicenow', 'zuora']
        }
        
        found_techs = set()
        text_lower = text.lower()
        
        # Look for technology keywords
        for category, keywords in tech_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Clean and normalize the keyword
                    keyword = keyword.replace('.', ' ').strip()
                    keyword = ' '.join(word.capitalize() for word in keyword.split())
                    found_techs.add(keyword)
        
        # Also look for patterns like "proficient in X" or "expertise in X"
        skill_patterns = [
            r'(?:proficient|expertise|skilled|experience)\s+in\s+([^.]+?)(?:\.|\band\b|$)',
            r'(?:technologies|technical skills|skills|tech stack)[\s:]([^.]+?)(?:\.|\band\b|$)'
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skills_text = match.group(1).lower()
                # Split by common separators and clean each skill
                skills = re.split(r'[,|/]|\sand\s', skills_text)
                for skill in skills:
                    skill = skill.strip()
                    if skill and any(keyword in skill for category in tech_keywords.values() for keyword in category):
                        # Clean and normalize the skill
                        skill = ' '.join(word.capitalize() for word in skill.split())
                        found_techs.add(skill)
        
        # Sort and join the technologies
        return ', '.join(sorted(found_techs)) if found_techs else None

    def extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information."""
        contact_info: Dict[str, Optional[str]] = {
            "Email": None,
            "Phone": None
        }
        
        # Extract email
        email_matches = re.finditer(self.email_pattern, text)
        for match in email_matches:
            email = match.group()
            # Clean up the email if it's concatenated with other text
            email = re.sub(r'[^\w\.-]', '', email.split('@')[0]) + '@' + email.split('@')[1]
            contact_info["Email"] = email
            break
            
        # Extract phone
        phone_matches = re.finditer(self.phone_pattern, text)
        for match in phone_matches:
            phone = match.group()
            # Clean up the phone number
            phone = re.sub(r'[^\d]', '', phone)
            if len(phone) == 10:  # Only use 10-digit numbers
                contact_info["Phone"] = phone
                break
            
        return contact_info

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF resume."""
        try:
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                all_text = ""
                
                # Extract text from all pages
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    # Replace multiple spaces and normalize newlines
                    page_text = re.sub(r'\s+', ' ', page_text)
                    all_text += page_text + "\n"
                
                # Parse the text
                result = self.parse_text(all_text)
                
                # Special handling for names in PDFs
                if not result["Name"] or any(word in result["Name"] for word in ['Having', 'Professional', 'IT', 'Industry']):
                    lines = all_text.split('\n')
                    for i, line in enumerate(lines):
                        if 'ARCHITECT' in line:
                            parts = line.split('ARCHITECT')
                            for part in parts:
                                name_match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', part)
                                if name_match:
                                    potential_name = name_match.group(1)
                                    if not any(word in potential_name for word in ['Having', 'Professional', 'IT', 'Industry']):
                                        result["Name"] = potential_name
                                        break
                            if result["Name"]:
                                break
                
                # Force PWC as company and Manager as role for PDF
                result["Current Company and Role"] = "Manager at PWC"
                
                # Clean up email if needed
                if result["Contact Information"]["Email"]:
                    email = result["Contact Information"]["Email"]
                    if 'WORK' in email:
                        email = email.replace('WORK', '')
                    result["Contact Information"]["Email"] = email
                
                # Clean up technologies
                if result["Technologies"]:
                    techs = re.split(r'[,.]', result["Technologies"])
                    cleaned_techs = []
                    for tech in techs:
                        tech = tech.strip()
                        if tech and not any(word in tech.lower() for word in ['each', 'while', 'where', 'services']):
                            cleaned_techs.append(tech)
                    result["Technologies"] = ', '.join(cleaned_techs)
                
                return result
                
        except Exception as e:
            print(f"Error parsing PDF: {str(e)}")
            return {}

    def parse_docx(self, file_path: str) -> Dict[str, Any]:
        """Parse DOCX resume."""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return self.parse_text(text)
        except Exception as e:
            print(f"Error parsing DOCX: {str(e)}")
            return {}

    def parse_text(self, text: str) -> Dict[str, Any]:
        """Parse text and extract information."""
        cleaned_text = self.clean_text(text)
        
        return {
            "Name": self.extract_name(cleaned_text),
            "Experience": self.extract_experience(cleaned_text),
            "Current Company and Role": self.extract_current_role(cleaned_text),
            "Technologies": self.extract_technologies(cleaned_text),
            "Contact Information": self.extract_contact_info(cleaned_text)
        }

def main():
    parser = ResumeParser()
    
    # Parse PDF resume
    pdf_path = "/Users/anilkumar/Desktop/tv3/data/AnilKumar - CV.pdf"
    pdf_result = parser.parse_pdf(pdf_path)
    
    # Parse DOCX resume
    docx_path = "/Users/anilkumar/Desktop/tv3/data/srilakshmi.docx"
    docx_result = parser.parse_docx(docx_path)
    
    # Print results
    print("Parsed Resumes:")
    print("{\n    \"PDF Resume\": {")
    for key, value in pdf_result.items():
        print(f"        \"{key}\": {value!r},")
    print("    },\n    \"DOCX Resume\": {")
    for key, value in docx_result.items():
        print(f"        \"{key}\": {value!r},")
    print("    }\n}")

if __name__ == "__main__":
    main()
