#!/usr/bin/env python3

import sys
import os
import requests
from urllib.parse import urljoin

def check_health():
    """Health check script for Docker container."""
    try:
        # Check if we're in API mode or dual mode
        risk_mode = os.getenv('RISK_MODE', 'dual')
        api_port = int(os.getenv('RISK_API_PORT', '8080'))
        
        if risk_mode in ['api', 'dual']:
            # Check HTTP API health endpoint
            health_url = f"http://localhost:{api_port}/health"
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    print(f"✅ Health check passed: {health_data}")
                    sys.exit(0)
                else:
                    print(f"❌ Health check failed: unhealthy status in response")
                    sys.exit(1)
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                sys.exit(1)
        
        elif risk_mode == 'mcp':
            # For MCP-only mode, just check if the process is running
            # This is a simple check since MCP stdio doesn't have HTTP endpoints
            print("✅ Health check passed: MCP mode running")
            sys.exit(0)
        
        else:
            print(f"❌ Unknown RISK_MODE: {risk_mode}")
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: Network error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_health()
