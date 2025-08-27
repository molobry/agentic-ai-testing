import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
import os

@dataclass
class TestStep:
    step_type: str  # 'given', 'when', 'then'
    description: str
    step_number: int

@dataclass
class BDDScenario:
    name: str
    steps: List[TestStep]
    file_path: str

class BDDParser:
    def __init__(self):
        pass
    
    def parse_scenario_file(self, file_path: str) -> BDDScenario:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Scenario file not found: {file_path}")
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            scenario_name = root.get('name', 'Unnamed Scenario')
            steps = []
            step_number = 1
            
            for element in root:
                if element.tag in ['given', 'when', 'then']:
                    step = TestStep(
                        step_type=element.tag,
                        description=element.text.strip() if element.text else '',
                        step_number=step_number
                    )
                    steps.append(step)
                    step_number += 1
            
            return BDDScenario(
                name=scenario_name,
                steps=steps,
                file_path=file_path
            )
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format in scenario file: {e}")
    
    def parse_scenarios_directory(self, directory: str) -> List[BDDScenario]:
        scenarios = []
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Scenarios directory not found: {directory}")
        
        for filename in os.listdir(directory):
            if filename.endswith('.xml'):
                file_path = os.path.join(directory, filename)
                try:
                    scenario = self.parse_scenario_file(file_path)
                    scenarios.append(scenario)
                except (FileNotFoundError, ValueError) as e:
                    print(f"Warning: Could not parse {filename}: {e}")
        
        return scenarios
    
    def validate_scenario(self, scenario: BDDScenario) -> List[str]:
        errors = []
        
        if not scenario.name:
            errors.append("Scenario must have a name")
        
        if not scenario.steps:
            errors.append("Scenario must have at least one step")
        
        has_given = any(step.step_type == 'given' for step in scenario.steps)
        has_when = any(step.step_type == 'when' for step in scenario.steps)
        has_then = any(step.step_type == 'then' for step in scenario.steps)
        
        if not has_given:
            errors.append("Scenario should have at least one 'given' step")
        if not has_when:
            errors.append("Scenario should have at least one 'when' step")
        if not has_then:
            errors.append("Scenario should have at least one 'then' step")
        
        for step in scenario.steps:
            if not step.description.strip():
                errors.append(f"Step {step.step_number} has empty description")
        
        return errors
    
    def print_scenario(self, scenario: BDDScenario):
        print(f"\nScenario: {scenario.name}")
        print("-" * 50)
        for step in scenario.steps:
            print(f"{step.step_number:2d}. {step.step_type.upper()}: {step.description}")
        print()

if __name__ == "__main__":
    parser = BDDParser()
    
    # Example usage
    try:
        scenario = parser.parse_scenario_file("scenarios/example.xml")
        parser.print_scenario(scenario)
        
        errors = parser.validate_scenario(scenario)
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✅ Scenario is valid!")
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")