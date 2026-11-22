@echo off
setlocal

set "DIR=%~dp0"
if not defined JAVA_HOME (
    if exist "C:\Program Files\Java\jdk-25" (
        set "JAVA_HOME=C:\Program Files\Java\jdk-25"
    ) else if exist "C:\Program Files\Java\jdk-25.0.2" (
        set "JAVA_HOME=C:\Program Files\Java\jdk-25.0.2"
    ) else if exist "C:\Program Files\Java\jdk-24" (
        set "JAVA_HOME=C:\Program Files\Java\jdk-24"
    )
)
set "MAVEN_HOME=%DIR%.tools\apache-maven-3.9.6"
if exist "%DIR%.m2\repository" (
    set "LOCAL_REPO=%DIR%.m2\repository"
) else (
    set "LOCAL_REPO=%USERPROFILE%\.m2\repository"
)

"%MAVEN_HOME%\bin\mvn.cmd" -Dmaven.repo.local="%LOCAL_REPO%" %*

