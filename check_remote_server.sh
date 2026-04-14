#!/bin/bash
# Check what's running on the remote server

echo "=== Checking remote server services ==="
echo ""

echo "1. Checking port 6006 (vLLM OpenAI API):"
ssh -p 20514 root@connect.westd.seetacloud.com "curl -s http://localhost:6006/health | head -5"
echo ""

echo "2. Checking port 6008 (Verification Server):"
ssh -p 20514 root@connect.westd.seetacloud.com "curl -s http://localhost:6008/health | head -5"
echo ""

echo "3. Checking running processes:"
ssh -p 20514 root@connect.westd.seetacloud.com "ps aux | grep -E '(vllm|api_server|uvicorn)' | grep -v grep"
echo ""

echo "4. Checking listening ports:"
ssh -p 20514 root@connect.westd.seetacloud.com "ss -tlnp | grep -E ':(6006|6008)'"
