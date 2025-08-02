#!/bin/bash

# Railway Deployment Script for AI at Risk Tournament
set -e

echo "🚂 AI at Risk Tournament - Railway Deployment Script"
echo "=================================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please log in to Railway..."
    railway login
fi

# Check if project is already initialized
if [ ! -f ".railway" ]; then
    echo "🚀 Initializing Railway project..."
    railway init
fi

# Set required environment variables
echo "🔧 Setting up environment variables..."

# Check if OPENAI_API_KEY is already set
if ! railway variables get OPENAI_API_KEY &> /dev/null; then
    echo "⚠️  OPENAI_API_KEY not found in Railway environment."
    echo "Please set it manually:"
    echo "  railway variables set OPENAI_API_KEY=your_key_here"
    echo ""
    echo "Or set it now:"
    read -p "Enter your OpenAI API key (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        railway variables set OPENAI_API_KEY="$api_key"
        echo "✅ OPENAI_API_KEY set successfully"
    fi
else
    echo "✅ OPENAI_API_KEY already configured"
fi

# Set tournament configuration (these have defaults in railway.toml but can be overridden)
echo "🎯 Tournament configuration:"
echo "  - Character submission phase: 15 minutes"
echo "  - Voting phase: 15 minutes" 
echo "  - Game phase: 1 hour"
echo "  - Results display: 5 minutes"
echo ""

read -p "Do you want to customize tournament timings? (y/N): " customize
if [[ $customize =~ ^[Yy]$ ]]; then
    read -p "Character submission duration (seconds, default 900): " submit_duration
    read -p "Voting phase duration (seconds, default 900): " voting_duration
    read -p "Game phase duration (seconds, default 3600): " game_duration
    read -p "Results display duration (seconds, default 300): " end_duration
    
    if [ ! -z "$submit_duration" ]; then
        railway variables set TOURNAMENT_SUBMIT_PHASE_DURATION="$submit_duration"
    fi
    if [ ! -z "$voting_duration" ]; then
        railway variables set TOURNAMENT_VOTING_PHASE_DURATION="$voting_duration"
    fi
    if [ ! -z "$game_duration" ]; then
        railway variables set TOURNAMENT_GAME_PHASE_DURATION="$game_duration"
    fi
    if [ ! -z "$end_duration" ]; then
        railway variables set TOURNAMENT_END_SCREEN_DURATION="$end_duration"
    fi
fi

# Deploy the application
echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "📊 Monitor your deployment:"
echo "  railway logs --tail"
echo ""
echo "🌐 Once deployed, your tournament will be available at:"
echo "  https://your-app.railway.app"
echo ""
echo "🔧 Useful commands:"
echo "  railway status          - Check deployment status"
echo "  railway logs           - View application logs"
echo "  railway variables      - Manage environment variables"
echo "  railway shell          - Connect to your deployment"
echo ""
echo "📚 For more help, see RAILWAY_DEPLOYMENT.md"
