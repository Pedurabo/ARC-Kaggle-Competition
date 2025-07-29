#!/usr/bin/env python3
"""
Submission Manager for ARC Prize 2025 Competition.
Helps manage daily submissions and track performance.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
from typing import Dict, List, Optional

class SubmissionManager:
    """Manages competition submissions and tracking."""
    
    def __init__(self, tracking_file: str = "submission_history.json"):
        self.tracking_file = Path(tracking_file)
        self.history = self._load_history()
    
    def _load_history(self) -> Dict:
        """Load submission history from file."""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {
            "submissions": [],
            "daily_count": {},
            "best_score": 0.0,
            "best_model": None,
            "team_info": {}
        }
    
    def _save_history(self):
        """Save submission history to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def can_submit_today(self) -> bool:
        """Check if we can submit today (1 per day limit)."""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = self.history["daily_count"].get(today, 0)
        return daily_count < 1
    
    def record_submission(self, model_name: str, submission_file: str, 
                         score: Optional[float] = None, notes: str = ""):
        """Record a submission in history."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        submission = {
            "date": today,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "file": submission_file,
            "score": score,
            "notes": notes
        }
        
        self.history["submissions"].append(submission)
        self.history["daily_count"][today] = self.history["daily_count"].get(today, 0) + 1
        
        if score and score > self.history["best_score"]:
            self.history["best_score"] = score
            self.history["best_model"] = model_name
        
        self._save_history()
        print(f"✅ Submission recorded: {model_name} ({submission_file})")
    
    def generate_submission(self, model_name: str, output_file: str) -> bool:
        """Generate a submission for a specific model."""
        if not self.can_submit_today():
            print("❌ Daily submission limit reached. Try again tomorrow.")
            return False
        
        print(f"🔄 Generating submission for {model_name}...")
        
        try:
            cmd = f"python src/main.py --model {model_name} --output {output_file}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            if Path(output_file).exists():
                self.record_submission(model_name, output_file, notes="Generated successfully")
                print(f"✅ Submission generated: {output_file}")
                return True
            else:
                print(f"❌ Submission file not created: {output_file}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating submission: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def get_submission_plan(self) -> List[Dict]:
        """Get the recommended submission plan."""
        models = [
            {"name": "baseline", "description": "Basic baseline for testing"},
            {"name": "pattern_matching", "description": "Pattern recognition approach"},
            {"name": "symbolic", "description": "Symbolic reasoning model"},
            {"name": "ensemble", "description": "Ensemble of multiple approaches"},
            {"name": "few_shot", "description": "Few-shot learning model"}
        ]
        
        plan = []
        for i, model in enumerate(models, 1):
            plan.append({
                "day": i,
                "model": model["name"],
                "description": model["description"],
                "file": f"submission_{model['name']}.json",
                "status": "pending"
            })
        
        return plan
    
    def show_status(self):
        """Show current submission status."""
        print("\n" + "=" * 60)
        print("📊 SUBMISSION STATUS")
        print("=" * 60)
        
        # Daily submission status
        today = datetime.now().strftime("%Y-%m-%d")
        daily_count = self.history["daily_count"].get(today, 0)
        can_submit = self.can_submit_today()
        
        print(f"📅 Today ({today}): {daily_count}/1 submissions used")
        print(f"🔄 Can submit today: {'✅ Yes' if can_submit else '❌ No'}")
        
        # Best performance
        if self.history["best_score"] > 0:
            print(f"🏆 Best score: {self.history['best_score']:.2f}% ({self.history['best_model']})")
        
        # Recent submissions
        recent_submissions = self.history["submissions"][-5:]
        if recent_submissions:
            print(f"\n📋 Recent submissions:")
            for sub in recent_submissions:
                score_str = f" ({sub['score']:.2f}%)" if sub['score'] else ""
                print(f"   {sub['date']}: {sub['model']}{score_str}")
        
        # Submission plan
        print(f"\n📋 Recommended submission plan:")
        plan = self.get_submission_plan()
        for item in plan[:3]:  # Show next 3 days
            print(f"   Day {item['day']}: {item['model']} - {item['description']}")
    
    def show_leaderboard_context(self):
        """Show leaderboard context and goals."""
        print("\n" + "=" * 60)
        print("🏆 LEADERBOARD CONTEXT")
        print("=" * 60)
        
        print("📊 Current Top Performers:")
        print("   1. Giotto.ai: 19.58%")
        print("   2. the ARChitects: 16.53%")
        print("   3. MindsAI @ Tufa Labs: 15.42%")
        print("   4. Guillermo Barbadillo: 11.94%")
        print("   5. rxe: 10.42%")
        
        print(f"\n🎯 Your Goals:")
        print(f"   🥉 Beat baseline (4.17%): {self.history['best_score']:.2f}%")
        print(f"   🥈 Reach competitive (10%): {'✅' if self.history['best_score'] >= 10 else '❌'}")
        print(f"   🥇 Target top tier (20%): {'✅' if self.history['best_score'] >= 20 else '❌'}")
        print(f"   👑 Grand prize (85%): {'✅' if self.history['best_score'] >= 85 else '❌'}")
        
        print(f"\n📈 Next Steps:")
        if self.history['best_score'] < 4.17:
            print("   1. Beat the 4.17% baseline")
        elif self.history['best_score'] < 10:
            print("   1. Reach 10% competitive performance")
        elif self.history['best_score'] < 20:
            print("   1. Target top 10 performance")
        else:
            print("   1. Aim for grand prize threshold")
        
        print("   2. Use daily submissions strategically")
        print("   3. Focus on novel reasoning approaches")
    
    def validate_submission(self, submission_file: str) -> bool:
        """Validate a submission file."""
        try:
            with open(submission_file, 'r') as f:
                submission = json.load(f)
            
            # Basic validation
            if not isinstance(submission, dict):
                print("❌ Submission must be a JSON object")
                return False
            
            for task_id, predictions in submission.items():
                if not isinstance(predictions, list):
                    print(f"❌ Task {task_id}: predictions must be a list")
                    return False
                
                for pred in predictions:
                    if not isinstance(pred, dict):
                        print(f"❌ Task {task_id}: prediction must be a dictionary")
                        return False
                    
                    if "attempt_1" not in pred or "attempt_2" not in pred:
                        print(f"❌ Task {task_id}: missing attempt_1 or attempt_2")
                        return False
            
            print("✅ Submission format is valid")
            return True
            
        except Exception as e:
            print(f"❌ Error validating submission: {e}")
            return False


def main():
    """Main function for submission manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARC Prize 2025 Submission Manager")
    parser.add_argument("--generate", type=str, help="Generate submission for model")
    parser.add_argument("--output", type=str, help="Output file for submission")
    parser.add_argument("--status", action="store_true", help="Show submission status")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard context")
    parser.add_argument("--validate", type=str, help="Validate submission file")
    parser.add_argument("--plan", action="store_true", help="Show submission plan")
    
    args = parser.parse_args()
    
    manager = SubmissionManager()
    
    if args.generate:
        output_file = args.output or f"submission_{args.generate}.json"
        manager.generate_submission(args.generate, output_file)
    
    elif args.status:
        manager.show_status()
    
    elif args.leaderboard:
        manager.show_leaderboard_context()
    
    elif args.validate:
        manager.validate_submission(args.validate)
    
    elif args.plan:
        print("\n📋 SUBMISSION PLAN")
        print("=" * 40)
        plan = manager.get_submission_plan()
        for item in plan:
            print(f"Day {item['day']}: {item['model']} - {item['description']}")
    
    else:
        # Show help
        print("🚀 ARC Prize 2025 - Submission Manager")
        print("=" * 50)
        print("\nUsage:")
        print("  python submission_manager.py --generate ensemble --output submission.json")
        print("  python submission_manager.py --status")
        print("  python submission_manager.py --leaderboard")
        print("  python submission_manager.py --validate submission.json")
        print("  python submission_manager.py --plan")
        
        # Show current status
        manager.show_status()


if __name__ == "__main__":
    main() 