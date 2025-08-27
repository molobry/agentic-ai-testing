#!/usr/bin/env python3

import asyncio
import sys
import json
import os
import argparse
from datetime import datetime
from typing import List, Dict

from src.bdd_parser import BDDParser, BDDScenario
from src.langchain_analyzer import LangChainConfig
from src.langgraph_orchestrator import TestOrchestrator

class LangChainTestRunner:
    def __init__(self, headless: bool = False, browser: str = "chromium", 
                 record_video: bool = False, ai_provider: str = "auto"):
        self.browser_config = {
            "headless": headless,
            "browser": browser,
            "record_video": record_video
        }
        
        # LangChain configuration
        self.langchain_config = LangChainConfig(
            provider=ai_provider,
            temperature=0.1,
            max_tokens=2000
        )
        
        self.parser = BDDParser()
        self.orchestrator = TestOrchestrator(self.langchain_config, self.browser_config)
        self.results = []
    
    async def run_scenario_file(self, scenario_file: str) -> Dict:
        """Run a single BDD scenario file using LangChain/LangGraph"""
        print(f"\n🎭 Starting LangChain test execution for: {scenario_file}")
        
        try:
            # Parse the scenario
            scenario = self.parser.parse_scenario_file(scenario_file)
            
            # Validate scenario
            validation_errors = self.parser.validate_scenario(scenario)
            if validation_errors:
                print("❌ Scenario validation failed:")
                for error in validation_errors:
                    print(f"   - {error}")
                return self._create_failure_result(scenario, validation_errors)
            
            # Execute using LangGraph orchestrator
            print(f"🧠 Using LangChain/LangGraph for intelligent test execution")
            result = await self.orchestrator.execute_scenario(scenario)
            
            # Enhance result with additional metadata
            result.update({
                "file_path": scenario_file,
                "framework": "LangChain + LangGraph",
                "ai_provider": self.langchain_config.provider,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to run scenario: {e}")
            return {
                "scenario_name": os.path.basename(scenario_file),
                "file_path": scenario_file,
                "success": False,
                "error": str(e),
                "steps": [],
                "execution_time": 0,
                "timestamp": datetime.now().isoformat(),
                "framework": "LangChain + LangGraph"
            }
    
    async def run_multiple_scenarios(self, scenario_files: List[str]) -> List[Dict]:
        """Run multiple scenario files"""
        results = []
        
        print(f"\n🚀 Starting LangChain test suite execution ({len(scenario_files)} scenarios)")
        
        for i, scenario_file in enumerate(scenario_files, 1):
            print(f"\n[{i}/{len(scenario_files)}] Processing: {scenario_file}")
            result = await self.run_scenario_file(scenario_file)
            results.append(result)
        
        # Print overall summary
        self._print_suite_summary(results)
        
        return results
    
    def _create_failure_result(self, scenario: BDDScenario, errors: List[str]) -> Dict:
        return {
            "scenario_name": scenario.name,
            "file_path": scenario.file_path,
            "success": False,
            "error": "; ".join(errors),
            "steps": [],
            "execution_time": 0,
            "timestamp": datetime.now().isoformat(),
            "framework": "LangChain + LangGraph"
        }
    
    def _print_suite_summary(self, results: List[Dict]):
        print("\n" + "=" * 80)
        print("🏁 LANGCHAIN TEST SUITE SUMMARY")
        print("=" * 80)
        
        total_scenarios = len(results)
        passed_scenarios = sum(1 for r in results if r["success"])
        failed_scenarios = total_scenarios - passed_scenarios
        
        total_execution_time = sum(
            r.get("execution_metadata", {}).get("total_execution_time", 0) 
            for r in results
        )
        
        print(f"📊 Total scenarios: {total_scenarios}")
        print(f"✅ Passed: {passed_scenarios}")
        print(f"❌ Failed: {failed_scenarios}")
        print(f"⏱️  Total execution time: {total_execution_time:.2f} seconds")
        print(f"🧠 AI Framework: LangChain + LangGraph")
        print(f"🎯 AI Provider: {self.langchain_config.provider}")
        
        if failed_scenarios > 0:
            print(f"\n❌ Failed scenarios:")
            for result in results:
                if not result["success"]:
                    print(f"   - {result['scenario_name']}: {result.get('error', 'Unknown error')}")
        
        print()
    
    def save_results_report(self, results: List[Dict], output_file: str = None):
        """Save detailed test results to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reports/langchain_test_results_{timestamp}.json"
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)
        
        report = {
            "test_suite": {
                "framework": "LangChain + LangGraph",
                "ai_provider": self.langchain_config.provider,
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": len(results),
                "passed_scenarios": sum(1 for r in results if r["success"]),
                "failed_scenarios": sum(1 for r in results if not r["success"]),
                "browser_config": self.browser_config,
                "langchain_config": {
                    "provider": self.langchain_config.provider,
                    "model_name": self.langchain_config.model_name,
                    "temperature": self.langchain_config.temperature,
                    "max_tokens": self.langchain_config.max_tokens
                }
            },
            "scenarios": results
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 LangChain test report saved: {output_file}")
        except IOError as e:
            print(f"❌ Failed to save report: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Agentic AI Web Testing Framework - LangChain Edition")
    parser.add_argument("scenario", nargs="+", help="BDD scenario file(s) to execute")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--browser", choices=["chromium", "firefox", "webkit"], 
                       default="chromium", help="Browser to use for testing")
    parser.add_argument("--record", action="store_true", help="Record video of test execution")
    parser.add_argument("--ai-provider", choices=["auto", "azure_openai", "anthropic", "openai"],
                       default="auto", help="AI provider for element analysis")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    # Initialize LangChain test runner
    runner = LangChainTestRunner(
        headless=args.headless, 
        browser=args.browser, 
        record_video=args.record,
        ai_provider=args.ai_provider
    )
    
    print("🧠 LangChain + LangGraph Agentic AI Testing Framework")
    print("=" * 60)
    print(f"AI Provider: {runner.langchain_config.provider}")
    print(f"Browser: {args.browser} {'(headless)' if args.headless else ''}")
    print(f"Video Recording: {'Enabled' if args.record else 'Disabled'}")
    
    try:
        if len(args.scenario) == 1:
            # Single scenario
            result = await runner.run_scenario_file(args.scenario[0])
            results = [result]
        else:
            # Multiple scenarios
            results = await runner.run_multiple_scenarios(args.scenario)
        
        # Save results
        runner.save_results_report(results, args.output)
        
        # Exit with appropriate code
        failed_count = sum(1 for r in results if not r["success"])
        sys.exit(failed_count)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ LangChain test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())