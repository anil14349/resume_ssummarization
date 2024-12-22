"""Chain-of-Thought enhancement for resume summarization."""
from typing import Dict, List

class ChainOfThoughtEnhancer:
    def __init__(self):
        """Initialize Chain-of-Thought enhancer."""
        self.reasoning_steps = [
            self._analyze_role,
            self._analyze_experience,
            self._analyze_achievements,
            self._analyze_skills,
            self._create_summary_outline
        ]

    def enhance_prompt(self, resume_data: Dict) -> str:
        """Enhance prompt with chain-of-thought reasoning."""
        thoughts = []
        for step in self.reasoning_steps:
            thought = step(resume_data)
            if thought:
                thoughts.append(thought)

        if not thoughts:
            return ""

        # Add thoughts as comments to guide but not appear in output
        prompt = "# Internal Analysis:\n"
        for thought in thoughts:
            prompt += f"# {thought}\n"
        
        prompt += "\nBased on the above analysis, create a professional summary that:\n"
        prompt += "1. Introduces the person and their current role\n"
        prompt += "2. Highlights their most impressive achievements\n"
        prompt += "3. Showcases their expertise and impact\n"
        prompt += "4. Maintains a professional and confident tone\n"
        prompt += "\nSummary:\n"
        
        return prompt

    def _analyze_role(self, data: Dict) -> str:
        """Analyze current role and responsibilities."""
        role = data.get("current_role", "")
        company = data.get("companies", "").split(",")[0].strip()
        if not role:
            return ""

        thought = "Role Analysis:\n"
        thought += f"- Position: {role}\n"
        if company:
            thought += f"- Company: {company}\n"
        thought += "- Key responsibilities:\n"
        
        if "HR" in role:
            thought += "  * Managing human resources processes\n"
            thought += "  * Handling employee relations\n"
            thought += "  * Ensuring compliance with regulations\n"
        elif "Engineer" in role:
            thought += "  * Developing technical solutions\n"
            thought += "  * Leading project implementations\n"
            thought += "  * Collaborating with cross-functional teams\n"
        
        return thought

    def _analyze_experience(self, data: Dict) -> str:
        """Analyze years of experience and career progression."""
        experience = data.get("years_experience", "")
        if not experience:
            return ""

        thought = "Experience Analysis:\n"
        thought += f"- Total experience: {experience} years\n"
        thought += "- Career level indicators:\n"
        thought += "  * Professional expertise\n"
        thought += "  * Leadership potential\n"
        thought += "  * Domain knowledge\n"
        
        return thought

    def _analyze_achievements(self, data: Dict) -> str:
        """Analyze key achievements and metrics."""
        achievements = data.get("achievements", "")
        if not achievements:
            return ""

        thought = "Achievement Analysis:\n"
        thought += "- Notable accomplishments:\n"
        
        # Extract metrics
        metrics = []
        if "%" in achievements:
            import re
            metrics = re.findall(r'(\d+%)', achievements)
        
        if metrics:
            thought += "- Quantifiable results:\n"
            for metric in metrics:
                thought += f"  * {metric} improvement\n"
        
        return thought

    def _analyze_skills(self, data: Dict) -> str:
        """Analyze technical skills and expertise."""
        skills = data.get("skills", "")
        if not skills:
            return ""

        thought = "Skills Analysis:\n"
        thought += "- Core competencies:\n"
        for skill in skills.split(",")[:5]:  # Top 5 skills
            thought += f"  * {skill.strip()}\n"
        
        return thought

    def _create_summary_outline(self, data: Dict) -> str:
        """Create an outline for the summary."""
        thought = "Summary Structure:\n"
        thought += "1. Professional Introduction\n"
        thought += "   - Name and current role\n"
        thought += "   - Years of experience\n"
        thought += "2. Key Achievements\n"
        thought += "   - Most significant metrics\n"
        thought += "   - Impact on organization\n"
        thought += "3. Professional Expertise\n"
        thought += "   - Core skills\n"
        thought += "   - Industry knowledge\n"
        
        return thought
