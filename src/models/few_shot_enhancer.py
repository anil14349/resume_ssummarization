"""Few-shot enhancer for resume summary generation."""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class FewShotEnhancer:
    """Few-shot enhancer for resume summary generation."""
    
    def __init__(self, base_model):
        """Initialize few-shot enhancer.
        
        Args:
            base_model: Base model to enhance
        """
        self.base_model = base_model
        self.examples = self._get_default_examples()
    
    def generate_summary(self, resume_data: Dict[str, Any]) -> str:
        """Generate a summary using few-shot learning.
        
        Args:
            resume_data: Dictionary containing parsed resume data
            
        Returns:
            Generated summary text
        """
        try:
            # Add examples to prompt
            prompt = self._build_few_shot_prompt(resume_data)
            
            # Generate summary using base model
            resume_data['prompt'] = prompt
            summary = self.base_model.generate_summary(resume_data)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating few-shot summary: {e}")
            raise
    
    def _build_few_shot_prompt(self, resume_data: Dict[str, Any]) -> str:
        """Build a few-shot prompt.
        
        Args:
            resume_data: Dictionary containing parsed resume data
            
        Returns:
            Few-shot prompt text
        """
        try:
            # Build example section
            examples = []
            for i, example in enumerate(self.examples, 1):
                example_text = f"Example {i}:\n"
                example_text += f"Name: {example['name']}\n"
                example_text += f"Current Role: {example['current_role']}\n"
                example_text += f"Years Experience: {example['years_experience']}\n"
                example_text += f"Companies: {', '.join(example['companies'])}\n"
                example_text += f"Skills: {', '.join(example['skills'])}\n"
                example_text += f"Achievements: {'; '.join(example['achievements'])}\n"
                example_text += f"Summary: {example['summary']}\n\n"
                examples.append(example_text)
            
            # Build target section
            target = f"Now generate a summary for:\n"
            target += f"Name: {resume_data['name']}\n"
            target += f"Current Role: {resume_data['current_role']}\n"
            target += f"Years Experience: {resume_data['years_experience']}\n"
            target += f"Companies: {', '.join(resume_data['companies'])}\n"
            target += f"Skills: {', '.join(resume_data['skills'])}\n"
            target += f"Achievements: {'; '.join(resume_data['achievements'])}\n"
            target += f"Summary:"
            
            # Combine examples and target
            prompt = "Here are some examples of professional summaries:\n\n"
            prompt += "\n".join(examples)
            prompt += target
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error building few-shot prompt: {e}")
            raise
    
    def _get_default_examples(self) -> List[Dict[str, Any]]:
        """Get default few-shot examples."""
        return [
            {
                'name': 'John Smith',
                'current_role': 'Senior Project Manager',
                'years_experience': 8,
                'companies': ['Tech Solutions Inc', 'Global Systems Ltd'],
                'skills': ['Project Management', 'Agile', 'Team Leadership', 'Risk Management', 'Stakeholder Communication'],
                'achievements': [
                    'Led $5M digital transformation project with 100% on-time delivery',
                    'Reduced project costs by 25% through process optimization',
                    'Managed cross-functional team of 15 members across 3 countries'
                ],
                'summary': 'Experienced Senior Project Manager with 8 years of expertise in leading complex technology projects. Proven track record of delivering multi-million dollar initiatives on time and under budget, including a successful $5M digital transformation project. Strong focus on process optimization, achieving 25% cost reduction through strategic improvements. Skilled in managing diverse, cross-functional teams and maintaining clear stakeholder communication.'
            },
            {
                'name': 'Sarah Johnson',
                'current_role': 'Marketing Director',
                'years_experience': 10,
                'companies': ['Brand Masters', 'Digital Marketing Pro'],
                'skills': ['Digital Marketing', 'Brand Strategy', 'Team Management', 'Content Strategy', 'Analytics'],
                'achievements': [
                    'Increased online engagement by 150% through targeted campaigns',
                    'Generated $2M in additional revenue via new marketing initiatives',
                    'Built and led high-performing marketing team of 12 specialists'
                ],
                'summary': 'Results-driven Marketing Director with a decade of experience in digital marketing and brand development. Successfully increased online engagement by 150% and generated $2M in additional revenue through innovative marketing strategies. Expert in building and leading high-performing teams, with proven success in developing and executing comprehensive marketing campaigns that drive business growth.'
            },
            {
                'name': 'Michael Chen',
                'current_role': 'Software Engineering Manager',
                'years_experience': 12,
                'companies': ['Tech Innovators', 'Software Solutions'],
                'skills': ['Software Development', 'Team Leadership', 'Agile Methodologies', 'System Architecture', 'Cloud Computing'],
                'achievements': [
                    'Reduced system downtime by 75% through infrastructure improvements',
                    'Led development of cloud platform serving 1M+ users',
                    'Mentored 20+ junior developers to senior positions'
                ],
                'summary': 'Seasoned Software Engineering Manager with 12 years of experience in leading high-impact development teams. Spearheaded critical infrastructure improvements resulting in 75% reduction in system downtime and led the development of a cloud platform serving over 1 million users. Strong focus on team development, successfully mentoring more than 20 junior developers to senior positions while maintaining technical excellence in software development and system architecture.'
            }
        ]
