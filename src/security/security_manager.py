"""
ARC AI System Security Manager
Comprehensive security with DevSecOps practices
"""

import hashlib
import hmac
import secrets
import jwt
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import json
import subprocess
import tempfile
from pathlib import Path

try:
    import bandit
    from bandit.core import manager
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False

try:
    import safety
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False

@dataclass
class SecurityEvent:
    """Security event record"""
    timestamp: datetime
    event_type: str
    severity: str
    description: str
    user: Optional[str] = None
    ip_address: Optional[str] = None
    resource: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ARCSecurityManager:
    """Comprehensive security manager for ARC AI system"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.logger = logging.getLogger(__name__)
        
        # Security configuration
        self.security_config = {
            'max_login_attempts': 5,
            'session_timeout': 3600,  # 1 hour
            'password_min_length': 12,
            'require_special_chars': True,
            'enable_rate_limiting': True,
            'enable_input_validation': True,
            'enable_output_sanitization': True
        }
        
        # Security state
        self.failed_login_attempts = {}
        self.active_sessions = {}
        self.security_events = []
        self.vulnerability_scan_results = {}
        
        # Initialize security measures
        self._init_security()
    
    def _init_security(self):
        """Initialize security measures"""
        # Create security directories
        security_dir = Path('security')
        security_dir.mkdir(exist_ok=True)
        
        # Generate security keys if not exists
        self._generate_security_keys()
        
        # Run initial security scan
        self.run_security_scan()
    
    def _generate_security_keys(self):
        """Generate security keys"""
        keys_file = Path('security/keys.json')
        
        if not keys_file.exists():
            keys = {
                'jwt_secret': secrets.token_hex(32),
                'api_key': secrets.token_hex(32),
                'encryption_key': secrets.token_hex(32),
                'hmac_key': secrets.token_hex(32)
            }
            
            with open(keys_file, 'w') as f:
                json.dump(keys, f, indent=2)
            
            self.logger.info("Security keys generated")
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        salt = secrets.token_hex(16)
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${hash_obj.hex()}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = hashed.split('$')
            hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hmac.compare_digest(hash_obj.hex(), hash_hex)
        except Exception:
            return False
    
    def generate_jwt_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'exp': datetime.utcnow() + timedelta(seconds=self.security_config['session_timeout']),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            self.log_security_event('token_expired', 'medium', f'Expired token for user')
            return None
        except jwt.InvalidTokenError:
            self.log_security_event('invalid_token', 'high', f'Invalid token provided')
            return None
    
    def validate_input(self, data: Any, input_type: str) -> bool:
        """Validate input data"""
        if not self.security_config['enable_input_validation']:
            return True
        
        # Basic input validation
        if isinstance(data, str):
            # Check for SQL injection patterns
            sql_patterns = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION']
            if any(pattern.lower() in data.lower() for pattern in sql_patterns):
                self.log_security_event('sql_injection_attempt', 'high', f'SQL injection attempt in {input_type}')
                return False
            
            # Check for XSS patterns
            xss_patterns = ['<script>', 'javascript:', 'onerror=', 'onload=']
            if any(pattern.lower() in data.lower() for pattern in xss_patterns):
                self.log_security_event('xss_attempt', 'high', f'XSS attempt in {input_type}')
                return False
        
        return True
    
    def sanitize_output(self, data: Any) -> Any:
        """Sanitize output data"""
        if not self.security_config['enable_output_sanitization']:
            return data
        
        if isinstance(data, str):
            # Basic HTML escaping
            data = data.replace('&', '&amp;')
            data = data.replace('<', '&lt;')
            data = data.replace('>', '&gt;')
            data = data.replace('"', '&quot;')
            data = data.replace("'", '&#x27;')
        
        return data
    
    def check_rate_limit(self, user_id: str, action: str) -> bool:
        """Check rate limiting"""
        if not self.security_config['enable_rate_limiting']:
            return True
        
        key = f"{user_id}:{action}"
        current_time = time.time()
        
        if key not in self.failed_login_attempts:
            self.failed_login_attempts[key] = []
        
        # Remove old attempts
        self.failed_login_attempts[key] = [
            t for t in self.failed_login_attempts[key] 
            if current_time - t < 300  # 5 minutes window
        ]
        
        # Check if too many attempts
        if len(self.failed_login_attempts[key]) >= 10:
            self.log_security_event('rate_limit_exceeded', 'medium', f'Rate limit exceeded for {user_id}')
            return False
        
        self.failed_login_attempts[key].append(current_time)
        return True
    
    def log_security_event(self, event_type: str, severity: str, description: str, 
                          user: str = None, ip_address: str = None, 
                          resource: str = None, details: Dict[str, Any] = None):
        """Log security event"""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            description=description,
            user=user,
            ip_address=ip_address,
            resource=resource,
            details=details
        )
        
        self.security_events.append(event)
        
        # Log based on severity
        if severity == 'high':
            self.logger.error(f"SECURITY ALERT: {description}")
        elif severity == 'medium':
            self.logger.warning(f"SECURITY WARNING: {description}")
        else:
            self.logger.info(f"SECURITY INFO: {description}")
    
    def run_security_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        scan_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'vulnerabilities': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Code security scan with Bandit
        if BANDIT_AVAILABLE:
            try:
                scan_results['code_scan'] = self._run_bandit_scan()
            except Exception as e:
                scan_results['warnings'].append(f"Bandit scan failed: {e}")
        
        # Dependency vulnerability scan
        if SAFETY_AVAILABLE:
            try:
                scan_results['dependency_scan'] = self._run_safety_scan()
            except Exception as e:
                scan_results['warnings'].append(f"Safety scan failed: {e}")
        
        # File permission check
        scan_results['file_permissions'] = self._check_file_permissions()
        
        # Environment security check
        scan_results['environment'] = self._check_environment_security()
        
        # Store results
        self.vulnerability_scan_results = scan_results
        
        # Save to file
        with open('security/scan_results.json', 'w') as f:
            json.dump(scan_results, f, indent=2)
        
        return scan_results
    
    def _run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security scan"""
        results = {
            'issues': [],
            'score': 0
        }
        
        try:
            # Run bandit on src directory
            cmd = ['bandit', '-r', 'src', '-f', 'json']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                bandit_results = json.loads(result.stdout)
                results['issues'] = bandit_results.get('results', [])
                results['score'] = bandit_results.get('metrics', {}).get('SEVERITY', {}).get('LOW', 0)
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _run_safety_scan(self) -> Dict[str, Any]:
        """Run Safety dependency scan"""
        results = {
            'vulnerabilities': [],
            'total_vulnerabilities': 0
        }
        
        try:
            # Run safety check
            cmd = ['safety', 'check', '--json']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                safety_results = json.loads(result.stdout)
                results['vulnerabilities'] = safety_results
                results['total_vulnerabilities'] = len(safety_results)
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _check_file_permissions(self) -> Dict[str, Any]:
        """Check file permissions"""
        results = {
            'issues': [],
            'secure_files': []
        }
        
        sensitive_files = [
            'security/keys.json',
            '.env',
            'configs/default.yaml'
        ]
        
        for file_path in sensitive_files:
            if os.path.exists(file_path):
                stat_info = os.stat(file_path)
                mode = stat_info.st_mode & 0o777
                
                if mode & 0o777 != 0o600:  # Should be user read/write only
                    results['issues'].append({
                        'file': file_path,
                        'current_permissions': oct(mode),
                        'recommended_permissions': '0o600'
                    })
                else:
                    results['secure_files'].append(file_path)
        
        return results
    
    def _check_environment_security(self) -> Dict[str, Any]:
        """Check environment security"""
        results = {
            'issues': [],
            'secure_settings': []
        }
        
        # Check for sensitive environment variables
        sensitive_vars = ['SECRET_KEY', 'DATABASE_PASSWORD', 'API_KEY']
        
        for var in sensitive_vars:
            if var in os.environ:
                value = os.environ[var]
                if len(value) < 32:
                    results['issues'].append({
                        'variable': var,
                        'issue': 'Weak secret (too short)'
                    })
                else:
                    results['secure_settings'].append(var)
        
        # Check for debug mode
        if os.getenv('DEBUG', 'False').lower() == 'true':
            results['issues'].append({
                'setting': 'DEBUG',
                'issue': 'Debug mode enabled in production'
            })
        
        return results
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        return {
            'scan_results': self.vulnerability_scan_results,
            'security_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'description': event.description,
                    'user': event.user,
                    'ip_address': event.ip_address
                }
                for event in self.security_events[-100:]  # Last 100 events
            ],
            'active_sessions': len(self.active_sessions),
            'failed_login_attempts': len(self.failed_login_attempts),
            'security_config': self.security_config
        }
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        # Simple encryption for demonstration
        # In production, use proper encryption libraries
        return hashlib.sha256(data.encode()).hexdigest()
    
    def validate_model_input(self, model_input: Any) -> bool:
        """Validate model input for security"""
        if isinstance(model_input, list):
            # Check for reasonable size limits
            if len(model_input) > 1000:
                self.log_security_event('input_size_limit', 'medium', 'Input size exceeds limit')
                return False
            
            # Check for reasonable values
            for item in model_input:
                if isinstance(item, (int, float)):
                    if abs(item) > 1e6:
                        self.log_security_event('input_value_limit', 'medium', 'Input value exceeds limit')
                        return False
        
        return True

# Global security manager instance
security_manager = ARCSecurityManager()

def get_security_manager() -> ARCSecurityManager:
    """Get the global security manager instance"""
    return security_manager 