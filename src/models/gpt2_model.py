"""GPT-2 model for resume summary generation."""
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import re
import random

from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


class GPT2Model:
    """GPT-2 model for generating resume summaries."""
    
    def __init__(self):
        """Initialize GPT-2 model."""
        logger.info("Initializing GPT-2 model")
        
        try:
            # Load model and tokenizer
            self.model_name = "gpt2"
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            
            # Set generation parameters
            self.max_length = 512  # Increased max length
            self.min_length = 100
            self.num_beams = 4
            self.temperature = 0.7
            self.top_p = 0.9
            self.top_k = 50
            
            logger.info("Successfully initialized GPT-2 model")
            
        except Exception as e:
            logger.error(f"Error initializing GPT-2 model: {e}")
            raise
    
    def generate_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a summary from the resume data."""
        try:
            name = resume_data.get('name', '')
            years = resume_data.get('years_experience', 0)
            skills = resume_data.get('skills', [])
            companies = resume_data.get('companies', [])
            achievements = resume_data.get('achievements', [])
            contact_info = resume_data.get('contact_info', {})
            
            # Build a basic summary starting with greeting
            summary = f"Hi, this is {name}"
            if years:
                summary += f" with {int(years)} years of experience"
            if skills:
                summary += f" specializing in {', '.join(skills[:5])}"
            summary += ". "
                
            if companies:
                summary += f"Currently working at {companies[0]}. "
                
            if achievements:
                summary += f"In my professional journey, {achievements[0]} "
                
            summary += "I am passionate about delivering exceptional results through innovative solutions."
            
            # Clean and return the summary
            return self._clean_summary(summary, contact_info)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary."
            
    def _detect_industry(self, resume_data: Dict[str, Any]) -> Tuple[str, float]:
        """Detect the industry from resume data with confidence score.
        
        Args:
            resume_data: Resume data dictionary
            
        Returns:
            Tuple of (industry, confidence_score)
        """
        # Combine relevant text for analysis
        text = ' '.join([
            resume_data.get('current_role', ''),
            ' '.join(resume_data.get('companies', [])),
            ' '.join(resume_data.get('skills', [])),
            ' '.join(resume_data.get('achievements', []))
        ]).lower()
        
        # Industry detection patterns
        industry_patterns = {
            'technology': {
                'primary': [
                    'software', 'developer', 'engineer', 'programming', 'tech',
                    'data scientist', 'devops', 'cloud', 'ai', 'machine learning'
                ],
                'secondary': [
                    'agile', 'scrum', 'git', 'aws', 'azure', 'python', 'java',
                    'javascript', 'react', 'node', 'database', 'api'
                ],
                'companies': [
                    'google', 'microsoft', 'amazon', 'apple', 'meta', 'ibm',
                    'oracle', 'salesforce', 'adobe', 'intel'
                ]
            },
            'healthcare': {
                'primary': [
                    'medical', 'healthcare', 'clinical', 'patient', 'health',
                    'nursing', 'pharmaceutical', 'biotech', 'telehealth'
                ],
                'secondary': [
                    'hospital', 'clinic', 'care', 'treatment', 'diagnosis',
                    'therapy', 'wellness', 'medical records', 'hipaa'
                ],
                'companies': [
                    'johnson & johnson', 'pfizer', 'unitedhealth', 'cvs health',
                    'anthem', 'cigna', 'humana', 'abbott', 'merck'
                ]
            },
            'finance': {
                'primary': [
                    'financial', 'banking', 'investment', 'trading', 'finance',
                    'accounting', 'audit', 'risk', 'compliance'
                ],
                'secondary': [
                    'portfolio', 'assets', 'securities', 'stocks', 'bonds',
                    'hedge fund', 'private equity', 'wealth management'
                ],
                'companies': [
                    'jpmorgan', 'goldman sachs', 'morgan stanley', 'blackrock',
                    'wells fargo', 'citi', 'bank of america', 'visa', 'mastercard'
                ]
            },
            'marketing': {
                'primary': [
                    'marketing', 'advertising', 'brand', 'digital marketing',
                    'content', 'social media', 'seo', 'growth', 'engagement'
                ],
                'secondary': [
                    'campaign', 'analytics', 'conversion', 'audience', 'crm',
                    'email marketing', 'ppc', 'content strategy'
                ],
                'companies': [
                    'wpp', 'omnicom', 'publicis', 'interpublic', 'dentsu',
                    'hubspot', 'mailchimp', 'nielsen', 'kantar'
                ]
            },
            'hr': {
                'primary': [
                    'hr', 'human resources', 'recruitment', 'talent',
                    'hiring', 'training', 'employee', 'workforce'
                ],
                'secondary': [
                    'benefits', 'compensation', 'onboarding', 'performance',
                    'culture', 'diversity', 'inclusion', 'labor relations'
                ],
                'companies': [
                    'adp', 'workday', 'linkedin', 'indeed', 'manpower',
                    'randstad', 'robert half', 'kelly services'
                ]
            },
            'education': {
                'primary': [
                    'education', 'teaching', 'training', 'curriculum',
                    'learning', 'student', 'academic', 'instruction'
                ],
                'secondary': [
                    'classroom', 'assessment', 'pedagogy', 'e-learning',
                    'educational technology', 'faculty', 'school'
                ],
                'companies': [
                    'pearson', 'mcgraw hill', 'chegg', 'coursera', 'udemy',
                    'blackboard', 'canvas', 'khan academy'
                ]
            },
            'creative': {
                'primary': [
                    'design', 'creative', 'art', 'ui', 'ux', 'visual',
                    'graphic', 'multimedia', 'content creation'
                ],
                'secondary': [
                    'typography', 'illustration', 'animation', 'branding',
                    'photography', 'video', 'adobe creative suite'
                ],
                'companies': [
                    'adobe', 'autodesk', 'pixar', 'dreamworks', 'weta',
                    'industrial light & magic', 'design studios'
                ]
            }
        }
        
        # Calculate industry scores
        scores = {}
        for industry, patterns in industry_patterns.items():
            score = 0
            
            # Check primary keywords (highest weight)
            for keyword in patterns['primary']:
                if keyword in text:
                    score += 3
            
            # Check secondary keywords (medium weight)
            for keyword in patterns['secondary']:
                if keyword in text:
                    score += 2
            
            # Check company names (lowest weight)
            for company in patterns['companies']:
                if company in text:
                    score += 1
            
            # Normalize score based on number of patterns
            total_patterns = (
                len(patterns['primary']) * 3 +
                len(patterns['secondary']) * 2 +
                len(patterns['companies'])
            )
            normalized_score = score / total_patterns if total_patterns > 0 else 0
            scores[industry] = normalized_score
        
        # Get industry with highest score
        if not scores:
            return 'general', 0.0
            
        max_industry = max(scores.items(), key=lambda x: x[1])
        return max_industry[0], max_industry[1]
    
    def _build_prompt(self, resume_data: Dict[str, Any]) -> str:
        """Build a prompt from resume data."""
        try:
            # Format achievements with better context
            achievements = resume_data.get('achievements', [])
            if achievements and isinstance(achievements[0], dict):
                # New format with detailed achievement data
                achievement = achievements[0]
                achievement_text = achievement.get('text', '')
                impact_level = achievement.get('impact_level', 'medium')
                categories = achievement.get('categories', {})
                primary_category = max(categories.items(), key=lambda x: x[1])[0] if categories else None
            else:
                # Legacy format or no achievements
                achievement_text = achievements[0] if achievements else ''
                impact_level = 'medium'
                primary_category = None
            
            # Format skills (take top 8)
            skills = resume_data.get('skills', [])[:8]
            
            # Detect industry with confidence score
            industry, confidence = self._detect_industry(resume_data)
            
            # Dynamic language templates based on industry and impact
            language_templates = {
                'technology': {
                    'high': {
                        'openings': [
                            "As an innovative technology professional specializing in",
                            "Leveraging cutting-edge expertise in",
                            "Pioneering technical solutions in",
                            "Driving technological innovation through expertise in",
                            "Architecting advanced solutions with proficiency in"
                        ],
                        'transitions': [
                            "Successfully delivered groundbreaking results by",
                            "Transformed technical landscapes through",
                            "Revolutionized systems by",
                            "Spearheaded major innovations including",
                            "Led enterprise-wide initiatives to"
                        ],
                        'closings': [
                            "committed to pushing the boundaries of technological innovation",
                            "focused on developing transformative technical solutions",
                            "dedicated to advancing technological excellence",
                            "passionate about leveraging technology to solve complex challenges",
                            "driven to pioneer next-generation solutions"
                        ]
                    },
                    'medium': {
                        'openings': [
                            "As a skilled technology professional with expertise in",
                            "Bringing technical proficiency in",
                            "Offering proven experience in",
                            "Combining technical knowledge with practical expertise in",
                            "Delivering effective solutions through proficiency in"
                        ],
                        'transitions': [
                            "Successfully implemented solutions by",
                            "Improved systems through",
                            "Enhanced capabilities by",
                            "Delivered valuable improvements including",
                            "Contributed to key initiatives to"
                        ],
                        'closings': [
                            "focused on delivering robust technical solutions",
                            "committed to technical excellence and innovation",
                            "dedicated to creating efficient technical solutions",
                            "passionate about solving technical challenges",
                            "driven to improve and optimize systems"
                        ]
                    }
                },
                'healthcare': {
                    'high': {
                        'openings': [
                            "As a dedicated healthcare professional specializing in",
                            "Bringing comprehensive expertise in",
                            "Delivering patient-centered excellence through",
                            "Leading healthcare innovations with focus on",
                            "Advancing medical care through expertise in"
                        ],
                        'transitions': [
                            "Transformed patient care by",
                            "Revolutionized healthcare delivery through",
                            "Significantly improved outcomes by",
                            "Led groundbreaking initiatives to",
                            "Spearheaded major improvements in"
                        ],
                        'closings': [
                            "committed to advancing patient care excellence",
                            "focused on improving healthcare outcomes",
                            "dedicated to healthcare innovation",
                            "passionate about enhancing patient experiences",
                            "driven to transform healthcare delivery"
                        ]
                    },
                    'medium': {
                        'openings': [
                            "As a healthcare professional with expertise in",
                            "Offering valuable experience in",
                            "Contributing to patient care through",
                            "Supporting healthcare delivery with focus on",
                            "Bringing healthcare expertise in"
                        ],
                        'transitions': [
                            "Improved patient care by",
                            "Enhanced healthcare services through",
                            "Contributed to better outcomes by",
                            "Supported key initiatives to",
                            "Helped implement improvements in"
                        ],
                        'closings': [
                            "focused on quality patient care",
                            "committed to healthcare excellence",
                            "dedicated to improving health outcomes",
                            "passionate about patient well-being",
                            "driven to enhance healthcare services"
                        ]
                    }
                },
                'hr': {
                    'high': {
                        'openings': [
                            "As a strategic HR leader specializing in",
                            "Driving organizational excellence through expertise in",
                            "Transforming workplace culture with focus on",
                            "Leading HR innovation with proficiency in",
                            "Spearheading people-first initiatives through"
                        ],
                        'transitions': [
                            "Transformed organizational culture by",
                            "Revolutionized HR practices through",
                            "Achieved exceptional results by",
                            "Led transformative initiatives to",
                            "Spearheaded major improvements in"
                        ],
                        'closings': [
                            "committed to fostering exceptional workplace cultures",
                            "focused on driving organizational excellence",
                            "dedicated to empowering organizational success",
                            "passionate about transforming workplace experiences",
                            "driven to optimize human capital strategies"
                        ]
                    },
                    'medium': {
                        'openings': [
                            "As an HR professional with expertise in",
                            "Bringing valuable experience in",
                            "Contributing to organizational success through",
                            "Supporting workforce excellence with focus on",
                            "Offering HR expertise in"
                        ],
                        'transitions': [
                            "Improved workplace practices by",
                            "Enhanced HR services through",
                            "Contributed to better outcomes by",
                            "Supported key initiatives to",
                            "Implemented improvements in"
                        ],
                        'closings': [
                            "focused on employee success",
                            "committed to workplace excellence",
                            "dedicated to organizational development",
                            "passionate about people management",
                            "driven to enhance HR practices"
                        ]
                    }
                }
            }
            
            # Get appropriate templates based on industry and impact
            templates = language_templates.get(industry, language_templates['hr'])
            impact_level = 'high' if confidence > 0.7 else 'medium'
            current_templates = templates[impact_level]
            
            # Build dynamic sections
            opening = random.choice(current_templates['openings'])
            transition = random.choice(current_templates['transitions'])
            closing = random.choice(current_templates['closings'])
            
            # Build structured prompt
            prompt = (
                "Generate a professional first-person summary following this structure:\n\n"
                "1. Strong Opening: Use this pattern:\n"
                f"{opening} [skills and expertise]\n\n"
                "2. Professional Journey: Use this transition:\n"
                f"{transition} [achievements and impact]\n\n"
                "3. Current Focus: End with this theme:\n"
                f"[current role and company], {closing}\n\n"
                "Make it engaging, confident, and authentic.\n"
                "Use natural language and avoid corporate jargon.\n\n"
                "Use this information:\n\n"
                f"Name: {resume_data.get('name', '')}\n"
                f"Current Role: {resume_data.get('current_role', '')}\n"
                f"Years Experience: {resume_data.get('years_experience', '')}\n"
                f"Companies: {', '.join(resume_data.get('companies', []))}\n"
                f"Key Skills: {', '.join(skills)}\n"
                f"Achievement: {achievement_text}\n"
                f"Industry: {industry}\n"
                f"Impact Level: {impact_level}\n\n"
                "Summary:\n"
            )
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise
    
    def _clean_metrics(self, text: str) -> str:
        """Clean and format metrics in text."""
        # Format percentages
        text = re.sub(r'\*(\d+(?:\.\d+)?)\*\s*%', r'\1%', text)
        text = re.sub(r'\*(\d+(?:\.\d+)?%)\*', r'\1', text)
        
        # Format currency
        text = re.sub(r'\*\$(\d+(?:\.\d+)?[KMB]?)\*', r'$\1', text)
        text = re.sub(r'\*(\d+(?:\.\d+)?)\*\s*(?=million|billion|thousand)', r'\1', text)
        
        # Format numbers
        text = re.sub(r'\*(\d+(?:\.\d+)?)\*\s*(?=times|x)', r'\1', text)
        text = re.sub(r'\*(\d+(?:\.\d+)?x)\*', r'\1', text)
        
        # Format ranges
        text = re.sub(r'\*(\d+)\*\s*-\s*\*(\d+)\*', r'\1-\2', text)
        text = re.sub(r'\*(\d+)\*\s*to\s*\*(\d+)\*', r'\1 to \2', text)
        
        return text

    def _clean_summary(self, summary: str, contact_info: Dict[str, str] = None) -> str:
        """Clean and format the generated summary."""
        try:
            # Extract greeting if present
            greeting_match = re.match(r'^Hi,\s+this\s+is\s+[^.]+\.?\s*', summary, re.I)
            greeting = greeting_match.group(0) if greeting_match else ''
            if greeting:
                summary = summary[len(greeting):].strip()
            
            # Split into sentences for better processing
            sentences = re.split(r'([.!?]+(?:\s+|$))', summary)
            cleaned_sentences = []
            
            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                
                # Skip sentences with contact information
                if any(pattern.search(sentence) for pattern in [
                    re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # Phone
                    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
                    re.compile(r'\b\d{5}(?:[-\s]\d{4})?\b'),  # ZIP
                    re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl)\b', re.I),  # Address
                    re.compile(r'\b(?:http[s]?://)?(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+\b'),  # URLs
                    re.compile(r'\b(?:linkedin\.com|github\.com|twitter\.com)/[\w-]+\b')  # Social media
                ]):
                    continue
                
                # Skip sentences that are too short or look like headers
                if len(sentence.split()) < 3 or all(word[0].isupper() for word in sentence.split() if word):
                    continue
                
                # Clean up formatting artifacts
                sentence = re.sub(r'\s+•\s+', ' ', sentence)  # Remove bullets
                sentence = re.sub(r'\s*\[\s*(?:skills and expertise|achievements and impact|current role and company)\s*\]\s*', '', sentence)
                
                # Clean metrics
                sentence = self._clean_metrics(sentence)
                
                # Fix spacing and punctuation
                sentence = re.sub(r'\s+([.,!?])', r'\1', sentence)
                sentence = re.sub(r'(?<=[.,!?])\s*(?=[A-Z])', ' ', sentence)
                sentence = re.sub(r'\s{2,}', ' ', sentence)
                
                # Clean up self-references
                sentence = re.sub(r'(?i)\b(?:my name is|i am called|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', r'\1', sentence)
                sentence = re.sub(r'(?i)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:speaking|here)\b', r'\1', sentence)
                
                # Improve transitions
                sentence = re.sub(r'(?i)\b(?:recently|in my current role|presently)\b', 
                               lambda m: random.choice(['In my current position', 'As part of my role', 'In my professional journey']), 
                               sentence)
                
                # Add personality
                sentence = re.sub(r'(?i)\b(?:try|trying|attempt|attempting)\s+to\b', 
                               lambda m: random.choice(['committed to', 'focused on', 'dedicated to', 'passionate about']), 
                               sentence)
                
                sentence = re.sub(r'(?i)\b(?:want|looking)\s+to\b', 
                               lambda m: random.choice(['eager to', 'excited to', 'motivated to', 'driven to']), 
                               sentence)
                
                cleaned_sentences.append(sentence.strip())
            
            # Combine cleaned sentences
            summary = ' '.join(cleaned_sentences)
            
            # Final formatting
            summary = summary.strip()
            if summary:
                summary = summary[0].upper() + summary[1:]
            
            # Ensure proper ending
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
            
            # Add engaging closing if missing
            if not any(phrase in summary.lower() for phrase in ['passionate about', 'committed to', 'dedicated to', 'driven to']):
                closing_statements = [
                    'I am passionate about delivering exceptional results through innovative solutions.',
                    'I am committed to driving continuous improvement and excellence.',
                    'I am dedicated to creating meaningful impact through strategic initiatives.',
                    'I am driven to achieve transformative outcomes through collaborative leadership.'
                ]
                summary += ' ' + random.choice(closing_statements)
            
            # Add contact information if available
            if contact_info:
                if contact_info.get('email') and contact_info.get('phone'):
                    summary += f" You can reach me at {contact_info['email']} or {contact_info['phone']}."
                elif contact_info.get('email'):
                    summary += f" You can reach me at {contact_info['email']}."
                elif contact_info.get('phone'):
                    summary += f" You can reach me at {contact_info['phone']}."
            
            # Add greeting back if present
            if greeting:
                summary = greeting + summary
            
            # Final cleanup
            summary = re.sub(r'\s+([.,!?])', r'\1', summary)  # Clean up spacing around punctuation
            summary = re.sub(r'(?<=[.,!?])\s*(?=[A-Z])', ' ', summary)  # Ensure space after punctuation
            summary = re.sub(r'\s{2,}', ' ', summary)  # Remove multiple spaces
            summary = summary.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error cleaning summary: {e}")
            return summary
    
    def _validate_summary(self, summary: str) -> bool:
        """Validate generated summary.
        
        Args:
            summary: Summary text to validate
            
        Returns:
            True if summary is valid, False otherwise
        """
        try:
            # Check length
            words = summary.split()
            if len(words) < 50 or len(words) > 200:
                return False
            
            # Check structure
            required_elements = [
                r'I am',  # First person intro
                r'\d+\s+years?\s+(?:of\s+)?experience',  # Experience mention
                r'expertise\s+(?:in|includes?)',  # Skills mention
                r'(?:achieved|accomplished|led|managed|developed|implemented|improved)',  # Achievement verb
                r'(?:currently\s+(?:work|am)|at|with)\s+[A-Z]'  # Company mention
            ]
            
            for pattern in required_elements:
                if not re.search(pattern, summary):
                    return False
            
            # Check for common issues
            if not summary.lower().startswith(('i ', 'as ')):
                return False
            
            if summary.count('.') < 3:  # At least three sentences
                return False
            
            # Check for proper first-person usage
            third_person = ['he ', 'she ', 'they ', 'their ', 'his ', 'her ']
            if any(term in summary.lower() for term in third_person):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating summary: {e}")
            raise
    
    def _generate_fallback_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a fallback summary if the main generation fails."""
        try:
            name = resume_data.get('name', '')
            years = resume_data.get('years_experience', 0)
            skills = resume_data.get('skills', [])
            companies = resume_data.get('companies', [])
            achievements = resume_data.get('achievements', [])
            contact_info = resume_data.get('contact_info', {})
            
            # Build a basic summary starting with greeting
            summary = f"Hi, this is {name}"
            if years:
                summary += f" with {int(years)} years of experience"
            if skills:
                summary += f" specializing in {', '.join(skills[:5])}"
            summary += ". "
                
            if companies:
                summary += f"Currently working at {companies[0]}. "
                
            if achievements:
                summary += f"In my professional journey, {achievements[0]} "
                
            summary += "I am passionate about delivering exceptional results through innovative solutions."
            
            # Clean and return the summary
            return self._clean_summary(summary, contact_info)
            
        except Exception as e:
            logger.error(f"Error generating fallback summary: {e}")
            return "Error generating summary."
