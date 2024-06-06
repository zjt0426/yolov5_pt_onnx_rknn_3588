#!/bin/bash

echo "CPU Usage:"
mpstat 1 1


echo -e "\nMemory Usage:"
free -h

echo -e "\nDisk I/O:"
iostat


echo -e "\nNetwork Usage:"
ifstat 1 1


echo -e "\nTemperature:"
sensors
