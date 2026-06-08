#!/usr/bin/env bash
set -euo pipefail

APP_NAME="STLViewer"
BUNDLE_ID="com.stlviewer.app"
INSTALL_DIR="/Applications"
APP_BUNDLE="$INSTALL_DIR/$APP_NAME.app"

echo "==> Building $APP_NAME (release)..."
swift build -c release

BINARY=".build/arm64-apple-macosx/release/$APP_NAME"

echo "==> Creating app bundle..."
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

cp "$BINARY" "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

cat > "$APP_BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>stl</string>
            </array>
            <key>CFBundleTypeName</key>
            <string>STL 3D Model</string>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            <key>LSHandlerRank</key>
            <string>Owner</string>
            <key>LSItemContentTypes</key>
            <array>
                <string>public.standard-tesselated-geometry-format</string>
            </array>
        </dict>
    </array>
    <key>UTImportedTypeDeclarations</key>
    <array>
        <dict>
            <key>UTTypeConformsTo</key>
            <array>
                <string>public.data</string>
            </array>
            <key>UTTypeDescription</key>
            <string>STL 3D Model</string>
            <key>UTTypeIdentifier</key>
            <string>public.standard-tesselated-geometry-format</string>
            <key>UTTypeTagSpecification</key>
            <dict>
                <key>public.filename-extension</key>
                <array>
                    <string>stl</string>
                </array>
            </dict>
        </dict>
    </array>
</dict>
</plist>
EOF

echo "==> Registering app with Launch Services..."
/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister -f "$APP_BUNDLE"

echo "==> Setting as default app for .stl files..."
if command -v duti &>/dev/null; then
    duti -s "$BUNDLE_ID" public.standard-tesselated-geometry-format all
    echo "    Done via duti."
else
    echo "    'duti' not found — installing via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install duti
        duti -s "$BUNDLE_ID" public.standard-tesselated-geometry-format all
        echo "    Done via duti."
    else
        echo "    WARNING: Could not set default app automatically."
        echo "    To finish: right-click any .stl file → Get Info → Open With → STLViewer → Change All"
    fi
fi

echo ""
echo "Done! $APP_BUNDLE installed."
echo "Run with: open /path/to/model.stl"
