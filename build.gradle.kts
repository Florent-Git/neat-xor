plugins {
    id("java")
    id("application")
    id("io.freefair.lombok") version "8.4"
}

group = "be.floshie.neat"
version = "1.0"

repositories {
    mavenCentral()
}

application {
    mainClass.set("be.floshie.neat.NeatMain");
}

dependencies {
    implementation("io.vavr:vavr:0.10.4")
    implementation("org.processing:core:3.3.7")
    implementation("info.picocli:picocli:4.7.5")
    implementation("org.slf4j:slf4j-api:1.7.30")
    implementation("org.slf4j:slf4j-simple:1.7.30")
    implementation("org.jgrapht:jgrapht-core:1.5.2")
    implementation("com.badlogicgames.ashley:ashley:1.7.4")
    implementation("com.fasterxml.jackson.dataformat:jackson-dataformat-yaml:2.16.1")


    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}