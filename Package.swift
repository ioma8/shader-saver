// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ShaderSaver",
    platforms: [
        .macOS(.v13),
    ],
    targets: [
        .executableTarget(
            name: "ShaderSaver"
        ),
        .testTarget(
            name: "ShaderSaverTests",
            dependencies: ["ShaderSaver"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
