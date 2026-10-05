#!/usr/bin/env python3
"""Offline developer build using already downloaded Maven jars; production uses Maven."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--java-home', type=Path, default=None)
parser.add_argument('--classpath-file', type=Path, default=None)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent
def run(argv):
    result = subprocess.run(argv)
    if result.returncode:
        raise SystemExit(result.returncode)
java_home = args.java_home or (Path(os.environ['JAVA_HOME']) if 'JAVA_HOME' in os.environ else None)
def tool(name):
    candidate = java_home / 'bin' / name if java_home else shutil.which(name)
    if not candidate:
        raise SystemExit(f'{name} was not found. Set JAVA_HOME to a JDK 21+ installation (ABBA 0.24 runs on Java 21).')
    return str(candidate)
classpath_file = args.classpath_file or root / 'target/test-classpath.txt'
if not classpath_file.is_file():
    raise SystemExit('Resolve the declared dependency classpath once: cd connectors/fiji && mvn dependency:build-classpath -Dmdep.outputFile=target/test-classpath.txt')
classpath = classpath_file.read_text().strip()
if not classpath:
    raise SystemExit('The Maven dependency classpath is empty.')
classes = root / 'target/classes'
# Start clean: a renamed or deleted class would otherwise stay in the JAR and the SciJava plugin index.
shutil.rmtree(classes, ignore_errors=True)
classes.mkdir(parents=True, exist_ok=True)
sources = sorted((root / 'src/main/java').rglob('*.java'))
# -proc:full: the SciJava annotation processor writes the plugin index (JDK 23+ no longer runs processors implicitly).
run([tool('javac'), '--release', '21', '-proc:full', '-encoding', 'UTF-8', '-cp', classpath, '-d', str(classes), *(str(p) for p in sources)])
# The same jar name Maven writes: artifactId-version from pom.xml.
pom = ElementTree.parse(root / 'pom.xml').getroot()
namespace = {'m': 'http://maven.apache.org/POM/4.0.0'}
name = pom.findtext('m:artifactId', namespaces=namespace) + '-' + pom.findtext('m:version', namespaces=namespace)
artifact = root / 'target' / (name + '.jar')
run([tool('jar'), 'cf', str(artifact), '-C', str(classes), '.'])
print(artifact)
if args.test:
    tests = root / 'target/test-classes'
    shutil.rmtree(tests, ignore_errors=True)
    tests.mkdir(parents=True, exist_ok=True)
    run([tool('javac'), '--release', '21', '-proc:none', '-encoding', 'UTF-8', '-cp', str(classes) + os.pathsep + classpath, '-d', str(tests), *(str(p) for p in (root / 'src/test/java').rglob('*.java'))])
    for smoke in ('ConnectorSmokeTest', 'HostGeometrySmokeTest'):
        run([tool('java'), '-Djava.awt.headless=true', '-Djava.util.prefs.userRoot=' + str(root / 'target/test-prefs'), '-cp', str(tests) + os.pathsep + str(classes) + os.pathsep + classpath, 'org.langslice.fiji.' + smoke])
