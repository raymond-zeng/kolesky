FROM quay.io/pypa/manylinux2014_x86_64:2024.10.07-1 AS builder

RUN yum update -y && \
        yum install -y python3 python3-pip && \
        yum clean all
RUN yum -y install python3-devel gcc