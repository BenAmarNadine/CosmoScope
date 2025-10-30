#!/bin/bash

sudo apt-get update -y

sudo apt-get install openjdk-8-jdk -y

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$JAVA_HOME/bin/:$PATH
export CLASS_PATH=$JAVA_HOME/lib:$CLASS_PATH
$JAVA_HOME/bin/javac -version

wget https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.4.1.tgz
tar -xvf apache-jmeter-5.4.1.tgz