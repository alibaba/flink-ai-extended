from flink:1.6-hadoop27

USER root

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /opt
# RUN cp opt/flink-table_2.11-1.6.0.jar lib/

# set up hadoop
RUN wget https://flink-ai-extended.oss-cn-beijing.aliyuncs.com/hadoop-2.8.0.tar.gz; tar xf hadoop-2.8.0.tar.gz; rm -f hadoop-2.8.0.tar.gz
ENV HADOOP_HDFS_HOME /opt/hadoop-2.8.0
ENV HADOOP_HOME /opt/hadoop-2.8.0
COPY core-site.xml /opt/hadoop-2.8.0/etc/hadoop/

# install flink 1.10
RUN rm -rf /opt/flink; wget https://flink-ai-extended.oss-cn-beijing.aliyuncs.com/flink-1.10.0-bin-scala_2.11.tgz
RUN tar xf flink-1.10.0-bin-scala_2.11.tgz
RUN mv flink-1.10.0 flink
RUN rm -f flink-1.10.0-bin-scala_2.11.tgz

#RUN rm -rf /opt/flink; wget http://etaose.oss-cn-hangzhou-zmf.aliyuncs.com/test/flink/package/flink-1.8-SNAPSHOT.tgz
#RUN tar xf flink-1.8-SNAPSHOT.tgz
#RUN mv flink-1.8-SNAPSHOT flink
#RUN rm -f flink-1.8-SNAPSHOT.tgz
RUN cp -r flink/opt/* flink/lib/

ENV HADOOP_CLASSPATH=/opt/hadoop-2.8.0/etc/hadoop:/opt/hadoop-2.8.0/share/hadoop/common/lib/*:/opt/hadoop-2.8.0/share/hadoop/common/*:/opt/hadoop-2.8.0/share/hadoop/hdfs:/opt/hadoop-2.8.0/share/hadoop/hdfs/lib/*:/opt/hadoop-2.8.0/share/hadoop/hdfs/*:/opt/hadoop-2.8.0/share/hadoop/yarn/lib/*:/opt/hadoop-2.8.0/share/hadoop/yarn/*:/opt/hadoop-2.8.0/share/hadoop/mapreduce/lib/*:/opt/hadoop-2.8.0/share/hadoop/mapreduce/*:/opt/hadoop-2.8.0/share/hadoop/common/:/opt/hadoop-2.8.0/share/hadoop/common/lib/:/opt/hadoop-2.8.0/share/hadoop/common/lib/:/opt/hadoop-2.8.0/share/hadoop/hdfs/:/opt/hadoop-2.8.0/share/hadoop/hdfs/lib/:/opt/hadoop-2.8.0/share/hadoop/yarn/:/opt/hadoop-2.8.0/share/hadoop/yarn/lib/:/opt/hadoop-2.8.0/contrib/capacity-scheduler/*.jar

# set up apt
COPY sources.list /etc/apt/
RUN apt-get update

# set up java
RUN apt-get --no-install-recommends install -y openjdk-8-jdk
ENV JAVA_HOME /docker-java-home

# install dev packages
RUN apt-get --no-install-recommends install -y apt-utils vim python2.7 python-pip zip net-tools procps
RUN apt-get --no-install-recommends install -y openjdk-8-dbg gdb git cmake python2.7-dbg
RUN apt-get --no-install-recommends install -y libz-dev
RUN mkdir /root/.pip
COPY pip.conf /root/.pip/
RUN pip install virtualenv

COPY bashrc /root/.bashrc
COPY vimrc /root/.vimrc
COPY docker-entrypoint.sh /

ENTRYPOINT ["/docker-entrypoint.sh"]

CMD ["help"]

ENV DEBIAN_FRONTEND teletype
