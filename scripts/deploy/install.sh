#!/bin/bash
cd /home/ec2-user/app
pip3.12 install -r requirements.txt --quiet 2>&1 | tail -3
