#!/bin/bash

killall -9 middlewared
sudo fuser -k 8080/tcp