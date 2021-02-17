#!/bin/bash
#set -x #echo on
#for city in Anaheim Massachusetts SiouxFalls NewYork
#for city in Anaheim Chicago GoldCoast Massachusetts NewYork SiouxFalls Sydney
for city in Anaheim Chicago Massachusetts NewYork SiouxFalls 
do
	echo "running $city"
	~/git/routing-framework-old/Build/Release/Launchers/./AssignTraffic -n 100 -i Locations/$city/edges.csv -od Locations/$city/od.csv -o Locations/$city/output_so -so
	~/git/routing-framework-old/Build/Release/Launchers/./AssignTraffic -n 100 -i Locations/$city/edges.csv -od Locations/$city/od.csv -o Locations/$city/output_ue
done
