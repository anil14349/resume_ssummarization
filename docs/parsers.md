# Parsers Documentation

## ParserFactory

The `ParserFactory` class manages the creation of different types of resume parsers.

### Class Definition
```python
class ParserFactory:
    def __init__(self):
        """Initialize parser factory with registered parser types."""
        self._parsers = {}
```

### Key Methods

#### create_parser
```python
def create_parser(self, parser_type: str) -> BaseParser:
    """Create and return appropriate parser instance.
    
    Args:
        parser_type (str): Type of parser to create
        
    Returns:
        BaseParser: Configured parser instance
        
    Raises:
        ValueError: If parser_type is not registered
    """
```

#### register_parser
```python
def register_parser(self, parser_type: str, parser_class: Type[BaseParser]):
    """Register new parser type.
    
    Args:
        parser_type (str): Name for the parser type
        parser_class (Type[BaseParser]): Parser class to register
    """
```

## BaseParser

The `BaseParser` class defines the interface for all resume parsers.

### Class Definition
```python
class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse resume content into structured format.
        
        Args:
            content (str): Raw resume content
            
        Returns:
            Dict[str, Any]: Structured resume data
        """
        pass
```

## ATSParser

The `ATSParser` class implements parsing optimized for ATS (Applicant Tracking Systems).

### Class Definition
```python
class ATSParser(BaseParser):
    def __init__(self, config: Optional[Dict] = None):
        """Initialize ATS parser.
        
        Args:
            config (Optional[Dict]): Parser configuration
        """
```

### Key Methods

#### parse
```python
def parse(self, content: str) -> Dict[str, Any]:
    """Parse resume content for ATS compatibility.
    
    Args:
        content (str): Raw resume content
        
    Returns:
        Dict[str, Any]: Structured resume data optimized for ATS
    """
```

#### extract_skills
```python
def extract_skills(self, text: str) -> List[str]:
    """Extract skills from text.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: Extracted skills
    """
```
