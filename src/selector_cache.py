import json
import os
import hashlib
from datetime import datetime
from typing import Optional, Dict, List

class SelectorCache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.ensure_cache_directory()
    
    def ensure_cache_directory(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_file_path(self, scenario_name: str) -> str:
        # Create a safe filename from scenario name
        safe_name = "".join(c for c in scenario_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_').lower()
        return os.path.join(self.cache_dir, f"{safe_name}_selectors.json")
    
    def _get_step_hash(self, step_description: str) -> str:
        # Create a hash of the step description for consistent lookup
        return hashlib.md5(step_description.encode()).hexdigest()[:8]
    
    def save_selector(self, scenario_name: str, step_description: str, selector: Dict):
        cache_file = self._get_cache_file_path(scenario_name)
        step_hash = self._get_step_hash(step_description)
        
        # Load existing cache or create new
        cache_data = self._load_cache_file(cache_file)
        
        # Update cache with new selector
        cache_entry = {
            "selector": selector,
            "step_description": step_description,
            "last_used": datetime.now().isoformat(),
            "success_count": cache_data.get("steps", {}).get(step_hash, {}).get("success_count", 0) + 1,
            "created_at": cache_data.get("steps", {}).get(step_hash, {}).get("created_at", datetime.now().isoformat())
        }
        
        if "steps" not in cache_data:
            cache_data["steps"] = {}
        
        cache_data["steps"][step_hash] = cache_entry
        cache_data["scenario_name"] = scenario_name
        cache_data["last_updated"] = datetime.now().isoformat()
        
        # Save updated cache
        self._save_cache_file(cache_file, cache_data)
        
        print(f"💾 Cached selector for step: {step_description[:50]}...")
    
    def get_selector(self, scenario_name: str, step_description: str) -> Optional[Dict]:
        cache_file = self._get_cache_file_path(scenario_name)
        step_hash = self._get_step_hash(step_description)
        
        cache_data = self._load_cache_file(cache_file)
        
        if "steps" not in cache_data or step_hash not in cache_data["steps"]:
            return None
        
        cache_entry = cache_data["steps"][step_hash]
        
        # Update last used timestamp
        cache_entry["last_used"] = datetime.now().isoformat()
        self._save_cache_file(cache_file, cache_data)
        
        print(f"🎯 Using cached selector for step: {step_description[:50]}...")
        return cache_entry["selector"]
    
    def _load_cache_file(self, cache_file: str) -> Dict:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load cache file {cache_file}: {e}")
                return {}
        return {}
    
    def _save_cache_file(self, cache_file: str, cache_data: Dict):
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save cache file {cache_file}: {e}")
    
    def get_scenario_cache_stats(self, scenario_name: str) -> Dict:
        cache_file = self._get_cache_file_path(scenario_name)
        cache_data = self._load_cache_file(cache_file)
        
        if "steps" not in cache_data:
            return {"total_steps": 0, "cache_hits": 0, "last_updated": None}
        
        total_steps = len(cache_data["steps"])
        total_successes = sum(step["success_count"] for step in cache_data["steps"].values())
        
        return {
            "total_steps": total_steps,
            "total_successes": total_successes,
            "last_updated": cache_data.get("last_updated"),
            "scenario_name": cache_data.get("scenario_name")
        }
    
    def clear_scenario_cache(self, scenario_name: str) -> bool:
        cache_file = self._get_cache_file_path(scenario_name)
        
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"🗑️ Cleared cache for scenario: {scenario_name}")
                return True
            except OSError as e:
                print(f"Error clearing cache: {e}")
                return False
        return False
    
    def list_cached_scenarios(self) -> List[Dict]:
        scenarios = []
        
        if not os.path.exists(self.cache_dir):
            return scenarios
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_selectors.json'):
                cache_file = os.path.join(self.cache_dir, filename)
                cache_data = self._load_cache_file(cache_file)
                
                if cache_data:
                    scenario_stats = {
                        "filename": filename,
                        "scenario_name": cache_data.get("scenario_name", "Unknown"),
                        "total_steps": len(cache_data.get("steps", {})),
                        "last_updated": cache_data.get("last_updated"),
                        "file_size": os.path.getsize(cache_file)
                    }
                    scenarios.append(scenario_stats)
        
        return sorted(scenarios, key=lambda x: x["last_updated"] or "", reverse=True)
    
    def export_cache_report(self, scenario_name: str) -> Dict:
        cache_file = self._get_cache_file_path(scenario_name)
        cache_data = self._load_cache_file(cache_file)
        
        if "steps" not in cache_data:
            return {"error": "No cache data found for scenario"}
        
        report = {
            "scenario_name": cache_data.get("scenario_name", scenario_name),
            "last_updated": cache_data.get("last_updated"),
            "total_steps": len(cache_data["steps"]),
            "steps": []
        }
        
        for step_hash, step_data in cache_data["steps"].items():
            step_report = {
                "step_description": step_data["step_description"],
                "selector_type": step_data["selector"]["type"],
                "selector_value": step_data["selector"]["value"],
                "success_count": step_data["success_count"],
                "created_at": step_data["created_at"],
                "last_used": step_data["last_used"]
            }
            report["steps"].append(step_report)
        
        # Sort steps by creation time
        report["steps"].sort(key=lambda x: x["created_at"])
        
        return report
    
    def optimize_cache(self, scenario_name: str, max_age_days: int = 30):
        """Remove old or unused cache entries"""
        cache_file = self._get_cache_file_path(scenario_name)
        cache_data = self._load_cache_file(cache_file)
        
        if "steps" not in cache_data:
            return
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Remove old entries
        original_count = len(cache_data["steps"])
        cache_data["steps"] = {
            step_hash: step_data
            for step_hash, step_data in cache_data["steps"].items()
            if datetime.fromisoformat(step_data["last_used"]) > cutoff_date
        }
        
        removed_count = original_count - len(cache_data["steps"])
        
        if removed_count > 0:
            cache_data["last_updated"] = datetime.now().isoformat()
            self._save_cache_file(cache_file, cache_data)
            print(f"🧹 Optimized cache: removed {removed_count} old entries")

if __name__ == "__main__":
    # Example usage
    cache = SelectorCache()
    
    # Save a selector
    selector = {"type": "id", "value": "login-button"}
    cache.save_selector("SauceDemo Login", "click login button", selector)
    
    # Retrieve a selector
    cached_selector = cache.get_selector("SauceDemo Login", "click login button")
    print(f"Retrieved: {cached_selector}")
    
    # Get stats
    stats = cache.get_scenario_cache_stats("SauceDemo Login")
    print(f"Cache stats: {stats}")
    
    # List all scenarios
    scenarios = cache.list_cached_scenarios()
    print(f"Cached scenarios: {scenarios}")
    
    # Export report
    report = cache.export_cache_report("SauceDemo Login")
    print(f"Cache report: {json.dumps(report, indent=2)}")