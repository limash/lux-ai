#!/usr/bin/env bash

for i in {1..100}
do
  echo "Cycle $i"
  python main.py
done

echo "The last cycle is $i"
