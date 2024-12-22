# Documentation

This directory contains detailed documentation for all components of the Resume Summary Generator system.

## Contents

- [Models](./models.md): Documentation for model implementations and factory
  - GPT2ResumeModel
  - EnhancedModelFactory
  
- [Parsers](./parsers.md): Documentation for resume parsing components
  - ParserFactory
  - BaseParser
  - ATSParser
  
- [Enhancers](./enhancers.md): Documentation for generation enhancers
  - FewShotEnhancer
  - ChainOfThoughtEnhancer
  
- [Evaluation](./evaluation.md): Documentation for evaluation components
  - MetricsCalculator
  - OutputEvaluator

## Class Relationships

```mermaid
classDiagram
    BaseResumeModel <|-- GPT2ResumeModel
    BaseResumeModel <|-- T5ResumeModel
    BaseResumeModel <|-- BARTResumeModel
    
    EnhancedModelFactory --> BaseResumeModel
    EnhancedModelFactory --> FewShotEnhancer
    EnhancedModelFactory --> ChainOfThoughtEnhancer
    
    BaseParser <|-- ATSParser
    ParserFactory --> BaseParser
    
    OutputEvaluator --> MetricsCalculator
    
    class BaseResumeModel {
        +generate_summary(text: str)
        +preprocess_text(text: str)
        +postprocess_output(output: str)
    }
    
    class EnhancedModelFactory {
        +create_model()
        +add_enhancement(type: str)
    }
    
    class BaseParser {
        +parse(content: str)
    }
    
    class OutputEvaluator {
        +evaluate_summary(generated: str, reference: str)
        +validate_output(summary: str)
    }
```
