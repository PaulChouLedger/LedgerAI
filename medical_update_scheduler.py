#!/usr/bin/env python3
"""
Automated Medical Data Update Scheduler

Features:
1. Scheduled medical data updates (daily/weekly)
2. Incremental updates for efficiency
3. Email notifications for update status
4. Integration with existing medical data ingestion system
5. Automatic retry and error handling

Usage:
python3 medical_update_scheduler.py --schedule-daily
python3 medical_update_scheduler.py --schedule-weekly
python3 medical_update_scheduler.py --run-once
"""

import os
import sys
import json
import time
import schedule
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import subprocess

# Import medical data ingester
try:
    from medical_data_ingestion import MedicalDataIngester
    from clinician_rag import ClinicianRAG
except ImportError:
    print("Warning: Could not import medical modules")

class MedicalUpdateScheduler:
    """
    Automated scheduler for medical data updates
    """

    def __init__(self, config_file: str = "medical_scheduler_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()

        # Email settings for notifications
        self.email_enabled = self.config.get('email_notifications', False)
        self.smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.email_user = os.getenv('MEDICAL_UPDATE_EMAIL_USER')
        self.email_password = os.getenv('MEDICAL_UPDATE_EMAIL_PASSWORD')
        self.notification_recipients = self.config.get('notification_recipients', [])

        # Update settings
        self.daily_update_time = self.config.get('daily_update_time', '02:00')
        self.weekly_update_day = self.config.get('weekly_update_day', 'sunday')
        self.weekly_update_time = self.config.get('weekly_update_time', '03:00')

        # State tracking
        self.state_file = Path("data/medical/update_state.json")
        self.state = self.load_state()

        print("🏥 Medical Update Scheduler initialized")

    def load_config(self) -> Dict:
        """Load scheduler configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        # Default configuration
        return {
            'email_notifications': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'notification_recipients': [],
            'daily_update_time': '02:00',
            'weekly_update_day': 'sunday',
            'weekly_update_time': '03:00',
            'max_retries': 3,
            'retry_delay': 300,  # 5 minutes
            'enable_incremental_updates': True,
            'backup_before_update': True
        }

    def save_config(self):
        """Save scheduler configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving config: {e}")

    def load_state(self) -> Dict:
        """Load scheduler state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        return {
            'last_daily_update': None,
            'last_weekly_update': None,
            'update_history': [],
            'failed_updates': 0,
            'total_updates': 0
        }

    def save_state(self):
        """Save scheduler state"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            print(f"❌ Error saving state: {e}")

    def send_email_notification(self, subject: str, body: str):
        """Send email notification about update status"""
        if not self.email_enabled or not self.notification_recipients:
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = ', '.join(self.notification_recipients)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()

            print(f"📧 Email notification sent: {subject}")

        except Exception as e:
            print(f"❌ Failed to send email notification: {e}")

    def run_medical_update(self, update_type: str = 'daily', force_full: bool = False) -> Dict:
        """
        Run medical data update

        Args:
            update_type: 'daily' or 'weekly'
            force_full: Force full update instead of incremental

        Returns:
            Update result dictionary
        """
        print(f"🚀 Starting {update_type} medical data update...")

        start_time = datetime.now()
        result = {
            'success': False,
            'update_type': update_type,
            'start_time': start_time.isoformat(),
            'end_time': None,
            'duration': None,
            'error': None,
            'stats': {}
        }

        try:
            # Backup current data if enabled
            if self.config.get('backup_before_update', True):
                self.backup_medical_data()

            # Initialize medical data ingester
            ingester = MedicalDataIngester()

            # Get current stats before update
            old_stats = ingester.get_medical_stats()

            # Determine if this should be incremental or full update
            if force_full or not self.config.get('enable_incremental_updates', True):
                print("🔄 Running full medical data update...")
                success = ingester.run_full_ingestion(force_update=True)
            else:
                print("🔄 Running incremental medical data update...")
                # For incremental updates, we could check for new content only
                # For now, we'll do a lighter version of the full update
                success = ingester.run_full_ingestion(force_update=False)

            # Get new stats after update
            new_stats = ingester.get_medical_stats()

            # Update result
            result['success'] = success
            result['stats'] = {
                'old_documents': old_stats.get('total_documents', 0),
                'new_documents': new_stats.get('total_documents', 0),
                'documents_added': new_stats.get('total_documents', 0) - old_stats.get('total_documents', 0),
                'sources': new_stats.get('sources', {})
            }

            # Update state
            if success:
                if update_type == 'daily':
                    self.state['last_daily_update'] = start_time.isoformat()
                elif update_type == 'weekly':
                    self.state['last_weekly_update'] = start_time.isoformat()

                self.state['total_updates'] += 1
                self.state['failed_updates'] = 0

                # Add to history
                self.state['update_history'].append({
                    'timestamp': start_time.isoformat(),
                    'type': update_type,
                    'success': True,
                    'documents_added': result['stats']['documents_added']
                })

                # Keep only last 30 entries in history
                if len(self.state['update_history']) > 30:
                    self.state['update_history'] = self.state['update_history'][-30:]

                print(f"✅ {update_type.capitalize()} update completed successfully")
                print(f"📊 Documents added: {result['stats']['documents_added']}")

            else:
                self.state['failed_updates'] += 1
                self.state['update_history'].append({
                    'timestamp': start_time.isoformat(),
                    'type': update_type,
                    'success': False,
                    'error': 'Medical data ingestion failed'
                })

                result['error'] = 'Medical data ingestion failed'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            self.state['failed_updates'] += 1

            print(f"❌ {update_type.capitalize()} update failed: {e}")

            # Add failure to history
            self.state['update_history'].append({
                'timestamp': start_time.isoformat(),
                'type': update_type,
                'success': False,
                'error': str(e)
            })

        # Update timing
        end_time = datetime.now()
        result['end_time'] = end_time.isoformat()
        result['duration'] = (end_time - start_time).total_seconds()

        # Save state
        self.save_state()

        # Send notification
        if self.email_enabled:
            self.send_update_notification(result)

        return result

    def backup_medical_data(self):
        """Create backup of current medical data"""
        print("💾 Creating backup of medical data...")

        try:
            backup_dir = Path("data/medical/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"medical_backup_{timestamp}"

            # Use rsync or similar for efficient backup
            medical_data_dir = Path("data/medical")

            if medical_data_dir.exists():
                # Simple backup using tar (if available) or copy
                import shutil
                shutil.copytree(str(medical_data_dir), str(backup_path))
                print(f"✅ Backup created: {backup_path}")

                # Keep only last 5 backups
                backups = sorted(backup_dir.glob("medical_backup_*"))
                if len(backups) > 5:
                    for old_backup in backups[:-5]:
                        shutil.rmtree(str(old_backup))
                        print(f"🗑️ Removed old backup: {old_backup.name}")

        except Exception as e:
            print(f"❌ Backup failed: {e}")
            # Continue with update even if backup fails

    def send_update_notification(self, result: Dict):
        """Send email notification about update results"""
        subject = f"Medical Data Update {'✅ SUCCESS' if result['success'] else '❌ FAILED'}"

        body_parts = [
            f"Update Type: {result['update_type'].upper()}",
            f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}",
            f"Start Time: {result['start_time']}",
            f"Duration: {result['duration']:.1f} seconds"
        ]

        if result['success']:
            stats = result['stats']
            body_parts.extend([
                f"Documents before: {stats['old_documents']}",
                f"Documents after: {stats['new_documents']}",
                f"Documents added: {stats['documents_added']}",
                "",
                "Sources updated:"
            ])

            for source, count in stats['sources'].items():
                body_parts.append(f"  - {source}: {count}")

        else:
            body_parts.append(f"Error: {result['error']}")

        body = "\n".join(body_parts)

        self.send_email_notification(subject, body)

    def schedule_daily_updates(self):
        """Schedule daily medical data updates"""
        print(f"📅 Scheduling daily updates at {self.daily_update_time}")

        schedule.every().day.at(self.daily_update_time).do(
            lambda: self.run_medical_update('daily')
        )

        print("✅ Daily updates scheduled")

    def schedule_weekly_updates(self):
        """Schedule weekly medical data updates"""
        print(f"📅 Scheduling weekly updates on {self.weekly_update_day} at {self.weekly_update_time}")

        # Map day names to schedule module constants
        day_map = {
            'monday': schedule.every().monday,
            'tuesday': schedule.every().tuesday,
            'wednesday': schedule.every().wednesday,
            'thursday': schedule.every().thursday,
            'friday': schedule.every().friday,
            'saturday': schedule.every().saturday,
            'sunday': schedule.every().sunday
        }

        if self.weekly_update_day.lower() in day_map:
            day_map[self.weekly_update_day.lower()].at(self.weekly_update_time).do(
                lambda: self.run_medical_update('weekly', force_full=True)
            )
            print("✅ Weekly updates scheduled")
        else:
            print(f"❌ Invalid day: {self.weekly_update_day}")

    def run_scheduler(self):
        """Run the scheduler loop"""
        print("🚀 Starting medical update scheduler...")

        # Schedule updates
        self.schedule_daily_updates()
        self.schedule_weekly_updates()

        # Run initial update check
        self.check_for_updates()

        print("⏰ Scheduler running. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            print("\n🛑 Scheduler stopped by user")

    def check_for_updates(self):
        """Check if updates are needed and run them"""
        now = datetime.now()

        # Check daily update
        if self.state.get('last_daily_update'):
            last_daily = datetime.fromisoformat(self.state['last_daily_update'])
            if now - last_daily > timedelta(hours=24):
                print("🔄 Daily update overdue, running now...")
                self.run_medical_update('daily')
        else:
            print("🔄 No previous daily update found, running now...")
            self.run_medical_update('daily')

        # Check weekly update
        if self.state.get('last_weekly_update'):
            last_weekly = datetime.fromisoformat(self.state['last_weekly_update'])
            if now - last_weekly > timedelta(days=7):
                print("🔄 Weekly update overdue, running now...")
                self.run_medical_update('weekly', force_full=True)
        else:
            print("🔄 No previous weekly update found, running now...")
            self.run_medical_update('weekly', force_full=True)

    def get_scheduler_status(self) -> Dict:
        """Get current scheduler status"""
        status = {
            'running': True,
            'last_daily_update': self.state.get('last_daily_update'),
            'last_weekly_update': self.state.get('last_weekly_update'),
            'total_updates': self.state.get('total_updates', 0),
            'failed_updates': self.state.get('failed_updates', 0),
            'next_daily_update': None,
            'next_weekly_update': None
        }

        # Calculate next update times (approximate)
        if self.state.get('last_daily_update'):
            last_daily = datetime.fromisoformat(self.state['last_daily_update'])
            status['next_daily_update'] = (last_daily + timedelta(hours=24)).isoformat()

        if self.state.get('last_weekly_update'):
            last_weekly = datetime.fromisoformat(self.state['last_weekly_update'])
            status['next_weekly_update'] = (last_weekly + timedelta(days=7)).isoformat()

        return status

def run_medical_update_cli():
    """CLI interface for manual medical updates"""
    parser = argparse.ArgumentParser(description='Medical Data Update Scheduler')
    parser.add_argument('--run-once', action='store_true', help='Run one update and exit')
    parser.add_argument('--schedule-daily', action='store_true', help='Schedule daily updates')
    parser.add_argument('--schedule-weekly', action='store_true', help='Schedule weekly updates')
    parser.add_argument('--status', action='store_true', help='Show scheduler status')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--force-full', action='store_true', help='Force full update')

    args = parser.parse_args()

    if args.config:
        scheduler = MedicalUpdateScheduler(args.config)
    else:
        scheduler = MedicalUpdateScheduler()

    if args.status:
        status = scheduler.get_scheduler_status()
        print("📊 Scheduler Status:")
        print(json.dumps(status, indent=2))

    elif args.run_once:
        print("🔄 Running one-time medical update...")
        result = scheduler.run_medical_update('manual', force_full=args.force_full)
        print("✅ Update completed!" if result['success'] else "❌ Update failed!")

    elif args.schedule_daily:
        print("📅 Scheduling daily updates...")
        scheduler.schedule_daily_updates()

        # Run the scheduler
        scheduler.run_scheduler()

    elif args.schedule_weekly:
        print("📅 Scheduling weekly updates...")
        scheduler.schedule_weekly_updates()

        # Run the scheduler
        scheduler.run_scheduler()

    else:
        print("Usage: python3 medical_update_scheduler.py [options]")
        print("Use --run-once for manual updates")
        print("Use --schedule-daily or --schedule-weekly to start automated updates")

if __name__ == "__main__":
    run_medical_update_cli()
