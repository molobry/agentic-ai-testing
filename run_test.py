#!/usr/bin/env python3

import asyncio
import sys
import json
import os
from datetime import datetime
from typing import List, Dict
import argparse

from src.bdd_parser import BDDParser, BDDScenario
from src.playwright_agent import PlaywrightAgent
from src.selector_cache import SelectorCache

class TestRunner:
    def __init__(self, headless: bool = False, browser: str = "chromium", record_video: bool = False):
        self.headless = headless
        self.browser = browser
        self.record_video = record_video
        self.parser = BDDParser()
        self.cache = SelectorCache()
        self.results = []
        
    async def run_scenario_file(self, scenario_file: str) -> Dict:
        """Run a single BDD scenario file"""
        print(f"\n🎭 Starting test execution for: {scenario_file}")
        
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
            
            # Execute the scenario
            result = await self.execute_scenario(scenario)
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
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_scenario(self, scenario: BDDScenario) -> Dict:
        """Execute a BDD scenario with Playwright"""
        start_time = datetime.now()
        
        # Initialize result structure
        result = {
            "scenario_name": scenario.name,
            "file_path": scenario.file_path,
            "success": False,
            "error": None,
            "steps": [],
            "execution_time": 0,
            "timestamp": start_time.isoformat(),
            "cache_stats": None
        }
        
        agent = None
        
        try:
            # Start browser automation agent
            agent = PlaywrightAgent(headless=self.headless, browser_type=self.browser, record_video=self.record_video)
            agent.set_current_scenario(scenario.name)  # Set scenario name before starting browser for video naming
            await agent.start_browser()
            
            print(f"\n📋 Executing scenario: {scenario.name}")
            print("=" * 60)
            
            # Execute each step
            for step in scenario.steps:
                print(f"\n{step.step_number}. {step.step_type.upper()}: {step.description}")
                
                # Handle navigation steps specially
                if self._is_navigation_step(step):
                    url = self._extract_url_from_step(step.description)
                    if url:
                        success = await agent.navigate_to(url)
                        step_result = {
                            "step_number": step.step_number,
                            "step_type": step.step_type,
                            "description": step.description,
                            "success": success,
                            "action_taken": f"navigate_to: {url}",
                            "screenshot": f"step_{step.step_number:03d}_navigate.png",
                            "error": None if success else "Navigation failed"
                        }
                    else:
                        step_result = {
                            "step_number": step.step_number,
                            "step_type": step.step_type,
                            "description": step.description,
                            "success": False,
                            "error": "Could not extract URL from navigation step"
                        }
                else:
                    # Execute regular step
                    step_result = await agent.execute_step(step.description, step.step_number)
                    step_result["step_type"] = step.step_type
                
                result["steps"].append(step_result)
                
                # Print step result
                if step_result["success"]:
                    print(f"   ✅ Success: {step_result.get('action_taken', 'completed')}")
                    if step_result.get('cached_selector_used'):
                        print("   🎯 Used cached selector")
                else:
                    print(f"   ❌ Failed: {step_result.get('error', 'Unknown error')}")
                    
                    # For 'then' steps (assertions), failure might be expected
                    if step.step_type != 'then':
                        print(f"   ⚠️  Stopping execution due to failed {step.step_type} step")
                        break
            
            # Determine overall success
            all_steps_successful = all(step["success"] for step in result["steps"])
            result["success"] = all_steps_successful
            
            # Get cache statistics
            result["cache_stats"] = self.cache.get_scenario_cache_stats(scenario.name)
            
            # Add video path to results if recording was enabled
            if self.record_video and agent and agent.video_path:
                result["video_recording"] = agent.video_path
                print(f"🎥 Video recording: {agent.video_path}")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ Scenario execution failed: {e}")
            
        finally:
            # Clean up
            if agent:
                await agent.close_browser()
            
            # Calculate execution time
            end_time = datetime.now()
            result["execution_time"] = (end_time - start_time).total_seconds()
            
            # Print summary
            self._print_scenario_summary(result)
            
        return result
    
    def _is_navigation_step(self, step) -> bool:
        description_lower = step.description.lower()
        return any(keyword in description_lower for keyword in [
            "navigate to", "go to", "visit", "open", "load", "browse to", "i am on"
        ])
    
    def _extract_url_from_step(self, description: str) -> str:
        import re
        
        # Look for URLs in quotes
        url_match = re.search(r'"(https?://[^"]+)"', description)
        if url_match:
            return url_match.group(1)
        
        url_match = re.search(r"'(https?://[^']+)'", description)
        if url_match:
            return url_match.group(1)
        
        # Look for URLs without quotes
        url_match = re.search(r'(https?://\S+)', description)
        if url_match:
            return url_match.group(1)
        
        return None
    
    def _create_failure_result(self, scenario: BDDScenario, errors: List[str]) -> Dict:
        return {
            "scenario_name": scenario.name,
            "file_path": scenario.file_path,
            "success": False,
            "error": "; ".join(errors),
            "steps": [],
            "execution_time": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def _print_scenario_summary(self, result: Dict):
        print("\n" + "=" * 60)
        print(f"📊 SCENARIO SUMMARY: {result['scenario_name']}")
        print("=" * 60)
        
        if result["success"]:
            print("✅ STATUS: PASSED")
        else:
            print("❌ STATUS: FAILED")
            if result.get("error"):
                print(f"   Error: {result['error']}")
        
        print(f"⏱️  Execution time: {result['execution_time']:.2f} seconds")
        print(f"📝 Total steps: {len(result['steps'])}")
        
        if result["steps"]:
            passed_steps = sum(1 for step in result["steps"] if step["success"])
            print(f"✅ Passed steps: {passed_steps}")
            print(f"❌ Failed steps: {len(result['steps']) - passed_steps}")
        
        # Cache statistics
        cache_stats = result.get("cache_stats")
        if cache_stats and cache_stats["total_steps"] > 0:
            print(f"🎯 Cache hits: {cache_stats['total_successes']} cached selectors used")
        
        # Video recording info
        if result.get("video_recording"):
            print(f"🎥 Video saved: {result['video_recording']}")
        
        print()
    
    async def run_multiple_scenarios(self, scenario_files: List[str]) -> List[Dict]:
        """Run multiple scenario files"""
        results = []
        
        print(f"\n🚀 Starting test suite execution ({len(scenario_files)} scenarios)")
        
        for i, scenario_file in enumerate(scenario_files, 1):
            print(f"\n[{i}/{len(scenario_files)}] Processing: {scenario_file}")
            result = await self.run_scenario_file(scenario_file)
            results.append(result)
        
        # Print overall summary
        self._print_suite_summary(results)
        
        return results
    
    def _print_suite_summary(self, results: List[Dict]):
        print("\n" + "=" * 80)
        print("🏁 TEST SUITE SUMMARY")
        print("=" * 80)
        
        total_scenarios = len(results)
        passed_scenarios = sum(1 for r in results if r["success"])
        failed_scenarios = total_scenarios - passed_scenarios
        
        total_execution_time = sum(r["execution_time"] for r in results)
        
        print(f"📊 Total scenarios: {total_scenarios}")
        print(f"✅ Passed: {passed_scenarios}")
        print(f"❌ Failed: {failed_scenarios}")
        print(f"⏱️  Total execution time: {total_execution_time:.2f} seconds")
        
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
            output_file = f"reports/test_results_{timestamp}.json"
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)
        
        report = {
            "test_suite": {
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": len(results),
                "passed_scenarios": sum(1 for r in results if r["success"]),
                "failed_scenarios": sum(1 for r in results if not r["success"]),
                "total_execution_time": sum(r["execution_time"] for r in results)
            },
            "scenarios": results
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 Test report saved: {output_file}")
        except IOError as e:
            print(f"❌ Failed to save report: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Agentic AI Web Testing Framework")
    parser.add_argument("scenario", nargs="+", help="BDD scenario file(s) to execute")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--browser", choices=["chromium", "firefox", "webkit"], 
                       default="chromium", help="Browser to use for testing")
    parser.add_argument("--record", action="store_true", help="Record video of test execution")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(headless=args.headless, browser=args.browser, record_video=args.record)
    
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
        print(f"❌ Test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())