#!/usr/bin/env sh

if [ "$1" != "" ]; then
    if [ ! -d "results" ]; then
	mkdir ./results
	echo "create results directory";
    fi 
    ./build/tools/caf