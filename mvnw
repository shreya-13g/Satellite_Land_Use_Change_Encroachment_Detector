#!/usr/bin/env sh
# POSIX maven wrapper
PRG="$0"
while [ -h "$PRG" ]; do
  ls=`ls -ld "$PRG"`
  link=`expr "$ls" : '.*-> \(.*\)$'`
  if expr "$link" : '/.*' > /dev/null; then
    PRG="$link"
  else
    PRG=`dirname "$PRG"`/"$link"
  fi
done
SAVED="`pwd`"
cd "`dirname \"$PRG\"`" >/dev/null
APP_HOME="`pwd`"
cd "$SAVED" >/dev/null

LOCAL_REPO="$APP_HOME/.m2/repository"

if [ -x "$APP_HOME/.tools/apache-maven-3.9.6/bin/mvn" ]; then
  exec "$APP_HOME/.tools/apache-maven-3.9.6/bin/mvn" -Dmaven.repo.local="$LOCAL_REPO" "$@"
else
  exec mvn -Dmaven.repo.local="$LOCAL_REPO" "$@"
fi
