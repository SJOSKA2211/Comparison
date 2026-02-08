#!/bin/bash
cd frontend
echo "Node Version: $(node -v)" > ../frontend_startup.log
echo "NPM Version: $(npm -v)" >> ../frontend_startup.log
npm run dev -- --port 3000 --host 0.0.0.0 >> ../frontend_startup.log 2>&1
