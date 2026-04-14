#!/bin/bash
sleep 5
curl -sf http://localhost:8000/api/v1/health > /dev/null
